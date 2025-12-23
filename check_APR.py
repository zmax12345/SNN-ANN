import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 引入你的模块
from dataset import SpeckleVoxelDataset

# ================= 配置区域 =================
FULL_CONFIG = {
    'files': {
        0.2: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        0.5: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        0.8: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        1.0: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        1.2: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        1.5: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        1.8: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ],
        2.0: [
            r'/data/zm/12.17_test/0.2mm_clip.csv'
        ]
    },
    'roi': {'row_start': 400, 'row_end': 499, 'col_start': 0, 'col_end': 1280},
    'window_size_ms': 25,  # 使用你当前的窗口大小
    'stride_ms': 25,  # 步长设大一点，跑得快，只需看分布
    'crop_size': 64
}


def analyze_apr():
    print("正在加载数据集进行 APR 统计...")
    # 我们只加载测试集部分文件来快速查看分布，或者你可以设 is_train=True 看全部
    dataset = SpeckleVoxelDataset(FULL_CONFIG, is_train=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    all_velocities = []
    all_aprs = []

    print("开始统计...")
    for _, labels, aprs in tqdm(loader):
        all_velocities.extend(labels.numpy().flatten())
        all_aprs.extend(aprs.numpy().flatten())

    all_velocities = np.array(all_velocities)
    all_aprs = np.array(all_aprs)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(all_velocities, all_aprs, alpha=0.1, s=10, color='purple')

    # 计算每个流速下的 APR 均值
    unique_v = np.unique(all_velocities)
    mean_aprs = []
    for v in unique_v:
        mean_apr = np.mean(all_aprs[all_velocities == v])
        mean_aprs.append(mean_apr)

    plt.plot(unique_v, mean_aprs, 'r-o', linewidth=2, label='Mean APR')

    plt.title('Physics Check: APR vs Velocity')
    plt.xlabel('Ground Truth Velocity (mm/s)')
    plt.ylabel('Active Pixel Ratio (APR)')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('/data/zm/12.22/check_apr_distribution.png', dpi=300)
    print("✅ 统计图已保存为 check_apr_distribution.png")

    # 打印关键统计数据
    print("\n=== APR 统计数据 ===")
    print(f"{'Velocity':<10} | {'Mean APR':<10} | {'Min APR':<10} | {'Max APR':<10}")
    print("-" * 50)
    for v in unique_v:
        mask = (all_velocities == v)
        aprs_v = all_aprs[mask]
        print(f"{v:<10.4f} | {np.mean(aprs_v):<10.4f} | {np.min(aprs_v):<10.4f} | {np.max(aprs_v):<10.4f}")


if __name__ == "__main__":
    analyze_apr()