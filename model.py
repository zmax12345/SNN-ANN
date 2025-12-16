import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer, surrogate


class SnnRegressor(nn.Module):
    def __init__(self, crop_size=64):
        """
        轻量级 SNN-ANN 混合回归网络
        输入: [Batch, T, 2, crop_size, crop_size]
        输出: [Batch, 1] (流速)
        """
        super().__init__()

        # --- 1. SNN Encoder (特征提取器) ---
        # 这一部分负责从含噪的时空数据中提取“运动特征”

        self.encoder = nn.Sequential(
            # Layer 1: 基础特征提取
            # 输入通道 2 (ON/OFF), 输出 32
            layer.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(32),  # BN层有助于抵抗不同光照带来的整体偏移
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # Pooling 1: 降噪关键步骤
            # 64x64 -> 32x32。这会融合 2x2 区域内的信息，平滑掉单个坏点
            layer.MaxPool2d(2, 2),

            # Layer 2: 提取更复杂的时空纹理
            layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # Pooling 2: 再次降噪与压缩
            # 32x32 -> 16x16
            layer.MaxPool2d(2, 2),

            # Layer 3: 高层语义
            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            # 最终尺寸: 128通道, 16x16 图像
        )

        # --- 2. ANN Decoder (回归器) ---
        # 这一部分负责根据特征的“强弱”读出流速

        # 展平后的维度: 128 * 16 * 16 = 32768
        # 为了轻量化，我们可以加一个全局平均池化 (Global Avg Pool)，大幅减少参数
        self.feature_map_size = 16
        self.flat_dim = 128  # 如果用GAP，维度就是通道数

        # 如果不用 GAP，参数量会很大，对于 2GB 显存可能吃紧
        # 这里我采用 Global Average Pooling 策略，既抗噪又轻量
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合，增强鲁棒性
            nn.Linear(64, 1)  # 输出标量流速
        )

    def forward(self, x_seq):
        # x_seq: [Batch, T, 2, H, W]
        # SpikingJelly 要求 T 在第一维: [T, Batch, 2, H, W]
        x_seq = x_seq.transpose(0, 1)

        # 1. SNN 前向传播
        # functional.multi_step_forward 会自动处理 T 维度
        # 输出 y_seq: [T, Batch, 128, 16, 16]
        y_seq = functional.multi_step_forward(x_seq, self.encoder)

        # 2. 时域聚合 (Temporal Aggregation)
        # 我们取 T 个时间步的平均发放率 (Mean Firing Rate)
        # 这一步把“时序”变成了“强度”，坏点产生的随机脉冲会被平均掉
        y_mean = y_seq.mean(0)  # [Batch, 128, 16, 16]

        # 3. 空域聚合 (Spatial Aggregation - GAP)
        # 将 16x16 的特征图平均成 1x1
        # 这意味着：只要 ROI 里有一部分区域反应强烈，整体均值就会有反应
        y_gap = self.gap(y_mean)  # [Batch, 128, 1, 1]
        y_flat = y_gap.view(y_gap.shape[0], -1)  # [Batch, 128]

        # 4. ANN 回归
        velocity = self.decoder(y_flat)

        # 5. 重置神经元状态 (必须!)
        functional.reset_net(self.encoder)

        return velocity