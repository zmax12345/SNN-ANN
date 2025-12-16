import numpy as np
import os

# 填入你任意一个生成好的 npy 文件路径
file_path = r'/data/zm/micro_dataset_shixiong/0.2778_1.npy'

if os.path.exists(file_path):
    data = np.load(file_path, mmap_mode='r')
    print(f"文件名: {file_path}")
    print(f"形状: {data.shape}")
    print(f"前5行数据:\n{data[:5]}")

    # 检查是否有 NaN
    if np.isnan(data).any():
        print("❌ 警告：数据中包含 NaN (空值)！这就是报错的原因。")
        # 看看是哪一列坏了
        for i, name in enumerate(['x', 'y', 'p', 't']):
            if np.isnan(data[:, i]).any():
                print(f"   -> 第 {i} 列 ({name}) 含有 NaN")
    else:
        print("✅ 数据中没有 NaN。")

    # 检查极性列的值范围
    p_col = data[:, 2]
    print(f"极性列 (P) 最小值: {np.nanmin(p_col)}, 最大值: {np.nanmax(p_col)}")
    if np.nanmin(p_col) < 0 or np.nanmax(p_col) > 1:
        print("❌ 警告：极性列包含除了 0 和 1 之外的数值！")
else:
    print("找不到文件")