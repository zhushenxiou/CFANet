import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 平方激活函数：用于模拟协方差矩阵特征
class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()

    def forward(self, x):
        return torch.square(x)

# 对数激活函数：用于数值稳定
class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

# 带权重约束的卷积层
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

# 特征提取层
class FeatureExtraction(nn.Module):
    def __init__(self,num_channels: int,F1=8, D=1, F2='auto',drop_out=0.25):
        super(FeatureExtraction, self).__init__()

        if F2 == 'auto':
            F2 = F1 * D

        # 长时频域特征分支（kernel_size=(1,125)）
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding='same', max_norm=2),
            nn.BatchNorm2d(F1),
        )

        # 短时频域特征分支（kernel_size=(1,15)）
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 15], padding='same', max_norm=2),
            nn.BatchNorm2d(F1),
        )

        # 分支1的通道卷积，深度可分离
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (num_channels, 1), padding=0, groups=F2, bias=False, max_norm=2),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid', max_norm=2),  # 点对点卷积
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1,32), stride=32),
            nn.Dropout(drop_out),
        )

        # 分支2的通道卷积
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[num_channels, 1], padding='valid', max_norm=2),
            nn.BatchNorm2d(F2),
            ActSquare(),  # 平方激活（模拟CSP空间滤波）
            nn.AvgPool2d((1,75), stride=25),
            ActLog(),  # 对数激活（稳定方差）
            nn.Dropout(drop_out),
        )

        # 全局平均池化：用于生成通道注意力权重
        self.globe1 = nn.AdaptiveAvgPool2d(1)
        self.globe2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 频域特征提取
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        # 通道特征提取
        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)

        # 池化提取
        pool_feature_1 = self.globe1(x_filter_1)
        pool_feature_2 = self.globe2(x_filter_2)

        return x_filter_1, x_filter_2, pool_feature_1, pool_feature_2

# 交叉特征增强模块
class CrossFeatureEnhancement(nn.Module):
    def __init__(self):
        super(CrossFeatureEnhancement, self).__init__()

    def forward(self, x_filter_1, x_filter_2, pool_feature_1, pool_feature_2):
        # 生成通道注意力权重
        weight_1 = F.softmax(pool_feature_1.view(pool_feature_1.size(0), -1), dim=-1)
        weight_1 = weight_1.view(pool_feature_1.size(0), -1, 1, 1)

        weight_2 = F.softmax(pool_feature_2.view(pool_feature_2.size(0), -1), dim=-1)
        weight_2 = weight_2.view(pool_feature_2.size(0), -1, 1, 1)

        # 特征交叉加权融合
        weight_feature_1 = x_filter_2 * weight_1
        weight_feature_2 = x_filter_1 * weight_2

        return weight_feature_1, weight_feature_2

# 改进的全局注意力模块
class GlobalAttention(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(GlobalAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # 将输入调整为 (batch_size, C, H * W) 的形式
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (batch_size, H * W, C)

        # 计算 Q, K, V
        q = self.w_q(x_flat).permute(0, 2, 1)  # (batch_size, C, H * W)
        k = self.w_k(x_flat).permute(0, 2, 1)  # (batch_size, C, H * W)
        v = self.w_v(x_flat).permute(0, 2, 1)  # (batch_size, C, H * W)

        # 正则化(增强数值稳定性）
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 注意力计算（缩放点积注意力）
        d_k = q.size(-1)
        attn_scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k)  # (batch_size, H * W, H * W)

        # 计算注意力权重
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 使用注意力权重加权 V
        attn_output = torch.matmul(attn_weights, v.transpose(-2, -1)).transpose(-2, -1).contiguous()  # (batch_size, C, H * W)

        # 还原形状为 (batch_size, C, H, W)
        x_attention = attn_output.view(B, C, H, W)

        # 跳跃连接
        x = x + self.dropout(x_attention)
        return x

# 基于卷积的局部注意力模块
class LocalAttention(nn.Module):
    def __init__(self, d_model, window_size=3, dropout=0.5):
        super(LocalAttention, self).__init__()
        self.window_size = window_size  # 窗口大小，用于局部注意力
        self.padding = window_size // 2  # 填充大小，使得卷积核可以覆盖局部区域

        # 线性投影层，仅用于计算 Q 和 K
        self.w_q = nn.Conv2d(d_model, d_model, kernel_size=window_size, padding=self.padding, bias=False)
        self.w_k = nn.Conv2d(d_model, d_model, kernel_size=window_size, padding=self.padding, bias=False)

        self.bn = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # 计算 Q 和 K，局部窗口的操作
        q = self.w_q(x)  # (batch_size, C, H, W)
        k = self.w_k(x)  # (batch_size, C, H, W)
        v = x  # 保持不变的 V

        # 正则化
        q = F.normalize(q, dim=1)  # 对通道进行正则化
        k = F.normalize(k, dim=1)

        # 计算注意力得分
        d_k = q.size(1)
        attn_scores = (q * k).sum(dim=1, keepdim=True) / math.sqrt(d_k)  # 点积方式计算注意力得分
        attn_weights = F.softmax(attn_scores, dim=1)  # 注意力权重

        # 使用注意力权重加权 V
        attn_output = attn_weights * v  # 仅使用注意力权重对 v 进行加权

        # 输出特征，使用 1x1 卷积进行通道压缩
        #out = self.output_conv(attn_output)

        # 跳跃连接
        out = x + self.dropout(self.bn(attn_output))
        return out

# 分类层
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(8, num_classes, (1, 69)), #2A and 2B: 69 HGD:78
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x

# CFANet模型
class CFANet(nn.Module):
    def __init__(self,num_classes: 4,num_channels: int,F1=8, D=1, drop_out=0.25):
        super(CFANet, self).__init__()

        self.featureExtraction = FeatureExtraction(num_channels, F1, D)
        self.crossFeatureEnhancement = CrossFeatureEnhancement()
        self.globalAttention = GlobalAttention(d_model=F1 * D)
        self.localAttention = LocalAttention(d_model=F1 * D)
        # drop减少过拟合
        self.drop = nn.Dropout(drop_out)
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # 特征提取
        x_filter_1, x_filter_2, pool_feature_1, pool_feature_2 = self.featureExtraction(x)
        # 特征交叉加权融合
        weight_feature_1, weight_feature_2 = self.crossFeatureEnhancement(x_filter_1, x_filter_2, pool_feature_1, pool_feature_2)

        # 全局+局部注意力机制
        x_attention1 = self.globalAttention(weight_feature_1)
        x_attention2 = self.localAttention(weight_feature_2)

        # 特征拼接：沿时间维度合并
        x_attention = torch.cat((x_attention1, x_attention2), 3)


        x = self.drop(x_attention)
        # 全连接分类层
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    block = CFANet(4, 22)
    input = torch.rand(16, 22, 1000)
    output = block(input)
    print(output.shape)
    summary(block.to('cuda'), input_size=(16, 22, 1000))