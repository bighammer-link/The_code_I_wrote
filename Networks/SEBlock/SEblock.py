import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Reference:https://arxiv.org/abs/1709.01507
    就是简单的降维之后再升维，计算出个通道对应的权重，返回输入和权重的乘积，也就是加权
    """

    def __init__(self, input_channel: int, ration: float):
        super().__init__()
        self.in_channel = input_channel
        self.scale_ration = ration
        self.reduce = nn.Conv2d(  # 1*1降维卷积
            in_channels=self.in_channel,
            out_channels=int(self.in_channel * self.scale_ration),  # 输出通道数是对输入通道数缩放ration倍数
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.expand = nn.Conv2d(  # 1*1 升维卷积
            in_channels=int(self.in_channel * self.scale_ration),  # 降维之后的通道数作为输入
            out_channels=self.in_channel,
            kernel_size=1,
            stride=1,
            bias=False
        )

    def forward(self, inputs: torch.Tensor):
        b, c, _, _ = inputs.size()
        x = F.adaptive_avg_pool2d(inputs, 1)  # 池化之后每个通道对应一个数值
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(b, c, 1, 1)  # 得到的x是权重向量
        return inputs * x



