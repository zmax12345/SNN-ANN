import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = "/data/zm/12.17_test/"

FILES = [
    "0.2mm_clip.csv", "0.5mm_clip.csv", "0.8mm_clip.csv",
    "1.0mm_clip.csv", "1.2mm_clip.csv", "1.5mm_clip.csv",
    "1.8mm_clip.csv", "2.0mm_clip.csv", "2.2mm_clip.csv",
    "2.5mm_clip.csv"
]

VELOCITIES = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]


def count_lines_and_get_times(filepath):
    """
    高效计算文件行数，并获取首尾时间戳
    """
    count = 0
    t_start = None
    t_end = None

    # 使用 chunksize 分块读取，既省内存又快
    # 我们只关心 't' (最后一列)，假设是第4列 (col, row, p, t)
    # 如果列数不对，pandas可能会报错，请根据实际情况调整 usecols
    try:
        # 先读第一行获取 t_start
        df_head = pd.read_csv(filepath, header=None, nrows=1)
        t_start = df_head.iloc[0, 3]  # 假设时间在第4列

        # 分块计数
        with pd.read_csv(filepath, header=None, chunksize=1000000, names=['c', 'r', 'p', 't']) as reader:
            for chunk in reader:
                count += len(chunk)
                # 更新 t_end (这是当前块的最后一行)
                t_end = chunk['t'].iloc[-1]

        return count, t_start, t_end

    except Exception as e:
        print(f"\n❌ 读取失败 {filepath}: {e}")
        return 0, 0, 0


def analyze_exact_density():
    exact_counts = []
    durations = []
    rates = []

    print(f"开始精确扫描 (可能需要几分钟)...")

    for filename in FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"❌ 警告: 找不到 {filename}")
            exact_counts.append(0)
            durations.append(0)
            rates.append(0)
            continue

        print(f"正在扫描: {filename} ...", end="\r")

        # === 核心：精确计数 ===
        N, t_start, t_end = count_lines_and_get_times(path)

        if N > 0:
            duration = (t_end - t_start) / 1e6  # 秒
            rate = N / duration
        else:
            duration = 0
            rate = 0

        exact_counts.append(N)
        durations.append(duration)
        rates.append(rate)

    print("\n扫描完成。")

    # --- 绘图 ---
    plt.figure(figsize=(15, 5))

    # 图1: 精确事件总数 (Total Count)
    plt.subplot(1, 3, 1)
    plt.plot(VELOCITIES, exact_counts, 'g-o', linewidth=2)
    plt.title("Exact Total Events (N)")
    plt.xlabel("Velocity (mm/s)")
    plt.ylabel("Total Lines")
    plt.grid(True, alpha=0.5)

    # 图2: 录制时长 (Duration)
    plt.subplot(1, 3, 2)
    plt.plot(VELOCITIES, durations, 'b-o', linewidth=2)
    plt.title("Recording Duration (T)")
    plt.xlabel("Velocity (mm/s)")
    plt.ylabel("Seconds")
    plt.grid(True, alpha=0.5)

    # 图3: 真实事件密度 (Rate = N/T)
    plt.subplot(1, 3, 3)
    plt.plot(VELOCITIES, rates, 'r-x', linewidth=2)
    plt.title("True Event Rate (N/T)")
    plt.xlabel("Velocity (mm/s)")
    plt.ylabel("Events per Second")
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig('/data/zm/12.22/check_exact_density.png', dpi=300)
    print("\n✅ 结果已保存为 check_exact_density.png")

    # 打印详细数据表
    print("\n" + "=" * 65)
    print(f"{'Vel':<6} | {'Total Events':<15} | {'Duration(s)':<12} | {'Rate(E/s)':<15}")
    print("-" * 65)
    for v, n, d, r in zip(VELOCITIES, exact_counts, durations, rates):
        print(f"{v:<6.1f} | {n:<15,d} | {d:<12.4f} | {r:<15,.1f}")
    print("=" * 65)


if __name__ == "__main__":
    analyze_exact_density()