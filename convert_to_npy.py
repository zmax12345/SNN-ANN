import numpy as np
import pandas as pd
import os

# 你的文件配置 (请补全路径)
CONFIG = {
    # 师兄微流控数据 (流速: 文件列表)
    'files': {
        0.2778: [
            r'/data/zm/micro_dataset_shixiong/0.2778_1.csv',
            r'/data/zm/micro_dataset_shixiong/0.2778_2.csv',
            r'/data/zm/micro_dataset_shixiong/0.2778_3.csv'
        ],
        0.5556: [
            r'/data/zm/micro_dataset_shixiong/0.5556_1.csv',
            r'/data/zm/micro_dataset_shixiong/0.5556_2.csv',
            r'/data/zm/micro_dataset_shixiong/0.5556_3.csv'
        ],
        0.8333: [
            r'/data/zm/micro_dataset_shixiong/0.8333_1.csv',
            r'/data/zm/micro_dataset_shixiong/0.8333_2.csv',
            r'/data/zm/micro_dataset_shixiong/0.8333_3.csv'
        ],
        1.1111: [
            r'/data/zm/micro_dataset_shixiong/1.1111_1.csv',
            r'/data/zm/micro_dataset_shixiong/1.1111_2.csv',
            r'/data/zm/micro_dataset_shixiong/1.1111_3.csv'
        ],
        1.3888: [
            r'/data/zm/micro_dataset_shixiong/1.3888_1.csv',
            r'/data/zm/micro_dataset_shixiong/1.3888_2.csv',
            r'/data/zm/micro_dataset_shixiong/1.3888_3.csv'
        ],
        1.6667: [
            r'/data/zm/micro_dataset_shixiong/1.6667_1.csv',
            r'/data/zm/micro_dataset_shixiong/1.6667_2.csv',
            r'/data/zm/micro_dataset_shixiong/1.6667_3.csv'
        ],
        1.9444: [
            r'/data/zm/micro_dataset_shixiong/1.9444_1.csv',
            r'/data/zm/micro_dataset_shixiong/1.9444_2.csv',
            r'/data/zm/micro_dataset_shixiong/1.9444_3.csv'
        ],
        2.2222: [
            r'/data/zm/micro_dataset_shixiong/2.2222_1.csv',
            r'/data/zm/micro_dataset_shixiong/2.2222_2.csv',
            r'/data/zm/micro_dataset_shixiong/2.2222_3.csv',
        ]
    },
}


def convert_csv_to_npy(csv_path):
    npy_path = csv_path.replace('.csv', '.npy')
    if os.path.exists(npy_path):
        print(f"Skipping {npy_path} (exists)")
        return

    print(f"Converting {csv_path} ...")

    chunk_size = 5000000
    chunks = []

    # 修正：指定读取 5 列
    # names 参数只是为了方便引用，pandas 会按顺序读取
    for chunk in pd.read_csv(csv_path, header=None,
                             names=['row', 'col', 'intensity', 'p', 't'],
                             chunksize=chunk_size):
        # 我们需要转换成标准格式: [x, y, p, t]
        # 注意：Row 是 Y, Col 是 X

        # 创建一个 4 列的数组 (我们暂时丢弃 intensity 以节省 20% 空间，
        # 如果你确实需要利用亮度值作为脉冲强度，可以改成 5 列)
        data = np.empty((len(chunk), 4), dtype=np.float32)

        data[:, 0] = chunk['col'].values  # X
        data[:, 1] = chunk['row'].values  # Y
        data[:, 2] = chunk['p'].values  # P
        data[:, 3] = chunk['t'].values  # T

        chunks.append(data)

    if len(chunks) > 0:
        full_data = np.vstack(chunks)
        np.save(npy_path, full_data)
        print(f"Saved: {npy_path} | Shape: {full_data.shape}")
    else:
        print(f"Warning: {csv_path} is empty!")


if __name__ == "__main__":
    for velocity, files in CONFIG['files'].items():
        for f in files:
            convert_csv_to_npy(f)