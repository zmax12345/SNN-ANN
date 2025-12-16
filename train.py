import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm
import numpy as np
import os

# 引入我们自己写的模块
from dataset import SpeckleVoxelDataset
from model import SnnRegressor

# ================= 配置区域 (Configuration) =================
# 请在这里填入你真实的 .csv 路径 (代码会自动去找同名的 .npy)
FULL_CONFIG = {
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
    'roi': {'row_start': 100, 'row_end': 200, 'col_start': 0, 'col_end': 1280},
    'window_size_ms': 10,  # 10ms 窗口
    'stride_ms': 5,  # 5ms 步长
    'crop_size': 64  # 输入尺寸 64x64
}

# 超参数
BATCH_SIZE = 64  # 如果显存够大(>8GB)，可以改到 64 或 128
LEARNING_RATE = 1e-3  # 初始学习率
NUM_EPOCHS = 50  # 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "/data/zm/12.16 qingliangji"


# ================= 训练与评估函数 =================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    # 使用 tqdm 显示进度条
    pbar = tqdm(loader, desc="Training", unit="batch")

    for inputs, labels in pbar:
        # inputs: [Batch, T, 2, 64, 64]
        # labels: [Batch, 1]
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(inputs)

        # 3. 计算损失
        loss = criterion(outputs, labels)

        # 4. 反向传播与更新
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()

        # 实时更新进度条上的 Loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_mse = 0.0
    total_samples = 0

    # 用于计算 RMSE (mm/s)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_mse += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 计算整体指标
    avg_mse = total_mse / total_samples

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))

    return avg_mse, rmse


# ================= 主程序 =================

def main():
    print(f"Using device: {DEVICE}")

    # 1. 准备数据
    print("Preparing Datasets...")
    # 训练集加载每个流速的前2个文件
    train_dataset = SpeckleVoxelDataset(FULL_CONFIG, is_train=True)
    # 测试集加载每个流速的第3个文件
    test_dataset = SpeckleVoxelDataset(FULL_CONFIG, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 2. 初始化模型
    model = SnnRegressor(crop_size=FULL_CONFIG['crop_size']).to(DEVICE)

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()  # 回归任务用均方误差
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率衰减策略 (可选：每15轮降低一半)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_rmse = float('inf')

    # 4. 开始训练循环
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 20)

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 验证
        val_mse, val_rmse = evaluate(model, test_loader, criterion, DEVICE)

        # 调整学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Train Loss (MSE): {train_loss:.5f}")
        print(f"Test RMSE: {val_rmse:.4f} mm/s  (Target: < 0.1)")
        print(f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_rmse < best_rmse:
            print(f"✅ New Best Model! (RMSE: {best_rmse:.4f} -> {val_rmse:.4f})")
            best_rmse = val_rmse
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            print(f"Current Best RMSE: {best_rmse:.4f}")

    print("\nTraining Complete!")
    print(f"Best RMSE on Test Set: {best_rmse:.4f} mm/s")
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()