import torch
import numpy as np
from torch.utils.data import Dataset
import os


class SpeckleVoxelDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.roi = config['roi']
        self.window_size_us = config['window_size_ms'] * 1000
        self.stride_us = config['stride_ms'] * 1000
        self.crop_size = config['crop_size']
        self.is_train = is_train

        self.samples = []

        # 这里的 cache 不再存数据，而是存 "mmap对象" (类似文件指针)
        # 占用内存极小
        self.mmap_files = []

        print(f"🚀 初始化数据集 ({'训练' if is_train else '测试'})...")

        global_file_idx = 0

        for velocity, file_list in config['files'].items():
            # 自动划分：前2个训练，后1个测试
            target_files = file_list[:-1] if is_train else file_list[-1:]

            for csv_path in target_files:
                # 自动寻找对应的 .npy 文件
                npy_path = csv_path.replace('.csv', '.npy')

                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"找不到 {npy_path}，请先运行转换脚本！")

                # --- 核心：内存映射加载 ---
                # 这行代码瞬间完成，不占内存
                events_mmap = np.load(npy_path, mmap_mode='r')

                self.mmap_files.append(events_mmap)
                current_cache_idx = len(self.mmap_files) - 1

                # 建立索引 (这一步需要读取 t 列，mmap 很快)
                print(f"   Indexing: {os.path.basename(npy_path)} (Shape: {events_mmap.shape})...")
                self.create_sliding_windows(current_cache_idx, events_mmap, velocity)

        print(f"✅ 加载完成！共 {len(self.samples)} 个样本。")

    def create_sliding_windows(self, cache_idx, events_mmap, label):
        # 我们只需要读取 't' 这一列来建立索引
        # mmap 允许我们只从磁盘读这一列
        times = events_mmap[:, 3]

        total_time = times[-1]
        curr_time = times[0]

        # 预计算所有窗口的起始时间，加速循环
        # 比如: start_times = [0, 5000, 10000, ...]
        start_times = np.arange(curr_time, total_time - self.window_size_us, self.stride_us)

        # 为了加速 __getitem__，我们需要预先找到每个窗口在数组中的 index (行号)
        # 使用 searchsorted 批量查找所有 start_time 对应的行号
        # 这可能需要几秒钟，但能极大加速后续训练
        start_indices = np.searchsorted(times, start_times)
        end_indices = np.searchsorted(times, start_times + self.window_size_us)

        for i in range(len(start_times)):
            idx_start = start_indices[i]
            idx_end = end_indices[i]

            # 只有当窗口内有数据时才加入
            if idx_end > idx_start:
                self.samples.append({
                    'cache_idx': cache_idx,
                    'idx_start': idx_start,  # 直接存行号，训练时不用再搜了
                    'idx_end': idx_end,
                    't_start': start_times[i],  # 存一下起始时间用于分桶
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]

        # 从磁盘读取一小块数据 (操作系统负责缓存)
        # events shape: [N_events, 4]
        file_idx = info['cache_idx']
        idx_start = info['idx_start']
        idx_end = info['idx_end']

        # 这里发生了真正的磁盘读取
        slice_events = self.mmap_files[file_idx][idx_start:idx_end]

        # 转为 Tensor
        # slice_events 还是 mmap 数组，复制一份到内存变成普通 tensor
        events_tensor = torch.from_numpy(np.array(slice_events)).float()

        # --- 空间 + 时间处理 ---
        voxel_grid = self.spatial_crop_and_voxelize(events_tensor, info['t_start'])

        return voxel_grid, torch.tensor([info['label']], dtype=torch.float32)

    def spatial_crop_and_voxelize(self, events, t_start):
        # 10ms -> 10 frames
        T = int(self.window_size_us / 1000)
        H_net, W_net = self.crop_size, self.crop_size
        grid = torch.zeros((T, 2, H_net, W_net), dtype=torch.float32)

        if len(events) == 0: return grid

        # 解析数据: x, y, p, t
        # events 已经是归一化后的数据了吗？
        # 注意：你在转换脚本里直接存了 x, y。
        # 如果 ROI 是 row 100-200，我们需要在裁剪前减去 offset

        x_raw = events[:, 0]
        y_raw = events[:, 1]
        # p = events[:, 2] # 暂存
        # t = events[:, 3] # 暂存

        # 这里的 y_raw 是 100-200。我们需要先归一化到 0-100
        y_norm = y_raw - self.roi['row_start']

        # 定义裁剪范围 (针对 0-100 的 y 和 0-1280 的 x)
        roi_h = self.roi['row_end'] - self.roi['row_start']  # 100
        roi_w = self.roi['col_end'] - self.roi['col_start']  # 1280

        if self.is_train:
            # 随机裁剪
            x_start = np.random.randint(0, roi_w - W_net + 1)
            y_start = np.random.randint(0, roi_h - H_net + 1)
        else:
            # 中心裁剪
            x_start = (roi_w - W_net) // 2
            y_start = (roi_h - H_net) // 2

        x_end = x_start + W_net
        y_end = y_start + H_net

        # 筛选
        # 注意：这里的 y_norm 已经减去了 100
        mask = (x_raw >= x_start) & (x_raw < x_end) & \
               (y_norm >= y_start) & (y_norm < y_end)

        # 应用掩码
        if not mask.any(): return grid

        valid_events = events[mask]

        # 最终坐标
        xs = (valid_events[:, 0] - x_start).long()
        ys = (valid_events[:, 1] - self.roi['row_start'] - y_start).long()
        ps = valid_events[:, 2].long()

        # 相对时间
        t_rel = valid_events[:, 3] - t_start
        t_idx = (t_rel / 1000).long()
        t_idx = torch.clamp(t_idx, 0, T - 1)

        # 填充
        grid[t_idx, ps, ys, xs] = 1.0

        return grid