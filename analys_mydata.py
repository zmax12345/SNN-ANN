import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ================= 配置区域：针对你的新数据 =================
# 你的数据路径 (根据你的截图 image_9313fc.png)
DATA_DIR = "/data/zm/12.17_test/"

# 你的文件名列表 (根据截图补充完整)
# 注意：文件名必须完全匹配
FILES = [
    "0.2mm_clip.csv",
    "0.5mm_clip.csv",
    "0.8mm_clip.csv",
    "1.0mm_clip.csv",
    "1.2mm_clip.csv",
    "1.5mm_clip.csv",
    "1.8mm_clip.csv",
    "2.0mm_clip.csv",
    "2.2mm_clip.csv",
    "2.5mm_clip.csv"
]

# 对应的真实流速标签
VELOCITIES = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]


def analyze_my_data():
    results_spread = []  # 空间覆盖率 (Unique Pixels / N)
    results_rate = []  # 事件产生率 (Events / sec)

    print(f"正在分析你的数据: {DATA_DIR}")

    for filename in FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"❌ 警告: 找不到文件 {path}")
            # 填个0占位，防止绘图错位
            results_spread.append(0)
            results_rate.append(0)
            continue

        print(f"分析中: {filename} ...")

        try:
            # --- 1. 读取数据 ---
            # 为了快，我们只读前 200万行 (这足够代表整体统计特性了)
            # 你的CSV很大(1.3GB)，全读会很慢且耗内存
            # 假设列顺序是: col, row, p, t (根据之前的经验)
            df = pd.read_csv(path, header=None, nrows=2000000)

            # 提取坐标和时间
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            t = df.iloc[:, 3].values

            # --- 2. 计算事件产生率 (Event Rate) ---
            # Rate = 事件数 / 时间跨度
            # 这是为了验证文件大小一致是否意味着密度失效
            duration_us = t[-1] - t[0]
            if duration_us > 0:
                rate = len(df) / (duration_us / 1e6)  # Events per second
            else:
                rate = 0

            # --- 3. 计算空间覆盖率 (Spatial Spread) ---
            # 看看同样的事件数量，到底覆盖了多少个不同的像素
            # 组合 x,y 为唯一ID (假设 x,y < 10000)
            coords = y.astype(np.int64) * 10000 + x.astype(np.int64)
            unique_pixels = len(np.unique(coords))

            # Spread Ratio = 覆盖的像素数 / 总事件数
            # 物理假设：流速快 -> 散斑跑得快 -> 覆盖面积大 -> Spread 高
            spread = unique_pixels / len(df)

            results_rate.append(rate)
            results_spread.append(spread)

        except Exception as e:
            print(f"❌ 读取错误: {e}")
            results_spread.append(0)
            results_rate.append(0)

    # --- 绘图对比 ---
    plt.figure(figsize=(12, 5))

    # 图1: 事件产生率 (对应你的文件大小)
    plt.subplot(1, 2, 1)
    plt.plot(VELOCITIES, results_rate, 'r-x', linewidth=2, markersize=8)
    plt.title("Check 1: Event Rate (Density)")
    plt.xlabel("Velocity (mm/s)")
    plt.ylabel("Events / Second")
    plt.grid(True, linestyle='--', alpha=0.5)
    # 如果这条线是平的，说明密度法彻底失效

    # 图2: 空间覆盖率 (我们新的希望)
    plt.subplot(1, 2, 2)
    plt.plot(VELOCITIES, results_spread, 'b-o', linewidth=2, markersize=8)
    plt.title("Check 2: Spatial Spread (Unique/Total)")
    plt.xlabel("Velocity (mm/s)")
    plt.ylabel("Spatial Spread Ratio")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('/data/zm/12.22/my_data_analysis.png', dpi=300)
    print("\n✅ 分析完成！结果已保存为 my_data_analysis.png")

    # 打印数值表
    print("\n" + "=" * 50)
    print(f"{'Velocity':<10} | {'Event Rate (E/s)':<20} | {'Spread Ratio':<20}")
    print("-" * 50)
    for v, r, s in zip(VELOCITIES, results_rate, results_spread):
        print(f"{v:<10.1f} | {r:<20.0f} | {s:<20.6f}")
    print("=" * 50)


if __name__ == "__main__":
    analyze_my_data()