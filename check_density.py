import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 用户配置区域 (User Configuration)
# ==========================================

file_path_low = '/data/zm/micro_dataset_shixiong/0.2778_1.csv'  # 低速数据路径
file_path_high = '/data/zm/micro_dataset_shixiong/2.2222_1.csv'  # 高速数据路径

# **在这里手动填写你要分析的像素坐标**
# 格式: (Row行号, Col列号)
# 你可以填一个，也可以填多个（会把结果叠加），建议先填一个最亮的点
target_pixels = [
    (103, 171),  # <--- 请修改这里，替换为你选择的坐标
    # (306, 412), # 如果想对比多个点，可以取消注释加一行
]

time_unit_scale = 1e6  # 时间戳单位 (1e6 = 微秒)
analyze_duration = 2.0  # 分析时长 (秒)


# ==========================================
# 2. 数据提取函数 (保持不变)
# ==========================================
def extract_isi(file_path, label, pixels):
    print(f"[{label}] 正在提取数据...")
    try:
        df = pd.read_csv(file_path, header=None, names=['row', 'col', 'brightness', 'polarity', 't'])
        df = df.set_index(['row', 'col']).sort_index()
        all_isi = []

        for (r, c) in pixels:
            if (r, c) in df.index:
                pixel_data = df.loc[(r, c)]
                if isinstance(pixel_data, pd.Series): pixel_data = pixel_data.to_frame().T

                t_raw = pixel_data['t'].values
                t_raw.sort()
                t_sec = (t_raw - t_raw[0]) / time_unit_scale
                t_valid = t_raw[t_sec <= analyze_duration]

                if len(t_valid) > 1:
                    isi_us = np.diff(t_valid)
                    isi_us = isi_us[isi_us > 0]
                    if len(isi_us) > 0:
                        all_isi.extend(np.log10(isi_us))
        return np.array(all_isi)
    except Exception as e:
        print(f"错误: {e}")
        return np.array([])


# ==========================================
# 3. 执行与绘图 (关键修改部分)
# ==========================================
isi_low = extract_isi(file_path_low, "Low Speed", target_pixels)
isi_high = extract_isi(file_path_high, "High Speed", target_pixels)

if len(isi_low) > 0 and len(isi_high) > 0:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 设定直方图的区间
    bins = np.linspace(2.5, 5.5, 100)  # 重点关注 Log 2.5~5.5 区域

    # 1. 计算直方图数据 (但不画图)
    counts_low, _ = np.histogram(isi_low, bins=bins, density=True)
    counts_high, _ = np.histogram(isi_high, bins=bins, density=True)

    # **关键步骤：获取两者的最大密度，统一Y轴上限**
    max_density = max(counts_low.max(), counts_high.max())
    y_limit = max_density * 1.1  # 留出 10% 顶空

    # 2. 绘制低速图
    ax1.hist(isi_low, bins=bins, color='#1f77b4', alpha=0.8, density=True)
    ax1.set_title(f'Low Speed (0.2778) ISI Distribution\nMax Density = {counts_low.max():.2f}')
    ax1.set_ylabel('Probability Density')
    ax1.set_ylim(0, y_limit)  # 锁定 Y 轴
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 3. 绘制高速图
    ax2.hist(isi_high, bins=bins, color='#d62728', alpha=0.8, density=True)
    ax2.set_title(f'High Speed (2.2222) ISI Distribution\nMax Density = {counts_high.max():.2f}')
    ax2.set_xlabel('Log10(ISI in microseconds) [3.0 = 1ms, 4.0 = 10ms]')
    ax2.set_ylabel('Probability Density')
    ax2.set_ylim(0, y_limit)  # 锁定 Y 轴 (与上面一致)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("数据不足。")