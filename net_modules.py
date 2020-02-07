import torch.nn as nn
import torch.nn.functional as F
import torch

def generate_nn_block():
    # 一层卷积
    # 一层 BatchNormalization num_features： 来自期望输入的特征数，该期望输入的大小为'batch_size x num_features depth x height x width
    model = nn.Sequential(
        nn.Conv2d(
            in_channels= 64,
            out_channels= 64,
            kernel_size= 3,
            stride = 1,
            padding=1
        ),
        nn.BatchNorm2d(64,affine=True) # 这里设为True表示我们希望学习BN层的 两个系数
    )
    return model


class PathNet(nn.Module):
    # 因为是19层重复的卷积层 所以我们要用自己写的生成模块函数（如上）
    def __init__(self):
        super(PathNet,self).__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(2,64,3,1,1),
            nn.BatchNorm2d(64)
        )

        self.hidden_block_1 = generate_nn_block()
        self.hidden_block_2 = generate_nn_block()
        self.hidden_block_3 = generate_nn_block()
        self.hidden_block_4 = generate_nn_block()
        self.hidden_block_5 = generate_nn_block()
        self.hidden_block_6 = generate_nn_block()
        self.hidden_block_7 = generate_nn_block()
        self.hidden_block_8 = generate_nn_block()
        self.hidden_block_9 = generate_nn_block()
        self.hidden_block_10 = generate_nn_block()
        self.hidden_block_11 = generate_nn_block()
        self.hidden_block_12 = generate_nn_block()
        self.hidden_block_13 = generate_nn_block()
        self.hidden_block_14 = generate_nn_block()
        self.hidden_block_15 = generate_nn_block()
        self.hidden_block_16 = generate_nn_block()
        self.hidden_block_17 = generate_nn_block()
        self.hidden_block_18 = generate_nn_block()
        self.hidden_block_19 = generate_nn_block()
        # 最后一层是输出一个channel 都是分值 那代表着我们需要把数值搞到0-1之间
        self.end_block = nn.Sequential(
            nn.Conv2d(64,1,3,1,1),
            nn.BatchNorm2d(1,affine=True)
        )
    # 前向传播包括20层卷积和一层线性输出
    def forward(self, x):
        x = F.relu(self.start_block(x))
        x = F.relu(self.hidden_block_1(x))
        x = F.relu(self.hidden_block_2(x))
        x = F.relu(self.hidden_block_3(x))
        x = F.relu(self.hidden_block_4(x))
        x = F.relu(self.hidden_block_5(x))
        x = F.relu(self.hidden_block_6(x))
        x = F.relu(self.hidden_block_7(x))
        x = F.relu(self.hidden_block_8(x))
        x = F.relu(self.hidden_block_9(x))
        x = F.relu(self.hidden_block_10(x))
        x = F.relu(self.hidden_block_11(x))
        x = F.relu(self.hidden_block_12(x))
        x = F.relu(self.hidden_block_13(x))
        x = F.relu(self.hidden_block_14(x))
        x = F.relu(self.hidden_block_15(x))
        x = F.relu(self.hidden_block_16(x))
        x = F.relu(self.hidden_block_17(x))
        x = F.relu(self.hidden_block_18(x))
        x = F.relu(self.hidden_block_19(x))
        output = self.end_block(x)
        return torch.clamp(output,0,1) # 限制在0-1 因为我们需要的而是相似度


class OverlapNet(nn.Module):
    def __init__(self):
        super(OverlapNet.self).__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        self.hidden_block_1 = generate_nn_block()
        self.hidden_block_2 = generate_nn_block()
        self.hidden_block_3 = generate_nn_block()
        self.hidden_block_4 = generate_nn_block()
        self.hidden_block_5 = generate_nn_block()
        self.hidden_block_6 = generate_nn_block()
        self.hidden_block_7 = generate_nn_block()
        self.hidden_block_8 = generate_nn_block()
        self.hidden_block_9 = generate_nn_block()
        self.hidden_block_10 = generate_nn_block()
        self.hidden_block_11 = generate_nn_block()
        self.hidden_block_12 = generate_nn_block()
        self.hidden_block_13 = generate_nn_block()
        self.hidden_block_14 = generate_nn_block()
        self.hidden_block_15 = generate_nn_block()
        self.hidden_block_16 = generate_nn_block()
        self.hidden_block_17 = generate_nn_block()
        self.hidden_block_18 = generate_nn_block()
        self.hidden_block_19 = generate_nn_block()
        # 最后一层是输出一个channel 利用sigmoid函数再辅助一个threshhold
        self.end_block = nn.Sequential(
            nn.Conv2d(64,1,3,1,1),
            nn.Sigmoid(),
            nn.Threshold(0.5,0) # 这里我们实际上为了能迭代快一些九八小于0.5的地方全部都变成0 如果能把大于0.5都变成1实际上更好 但是还没找到办法
        )

    def forward(self,x):
        x = F.relu(self.start_block(x))
        x = F.relu(self.hidden_block_1(x))
        x = F.relu(self.hidden_block_2(x))
        x = F.relu(self.hidden_block_3(x))
        x = F.relu(self.hidden_block_4(x))
        x = F.relu(self.hidden_block_5(x))
        x = F.relu(self.hidden_block_6(x))
        x = F.relu(self.hidden_block_7(x))
        x = F.relu(self.hidden_block_8(x))
        x = F.relu(self.hidden_block_9(x))
        x = F.relu(self.hidden_block_10(x))
        x = F.relu(self.hidden_block_11(x))
        x = F.relu(self.hidden_block_12(x))
        x = F.relu(self.hidden_block_13(x))
        x = F.relu(self.hidden_block_14(x))
        x = F.relu(self.hidden_block_15(x))
        x = F.relu(self.hidden_block_16(x))
        x = F.relu(self.hidden_block_17(x))
        x = F.relu(self.hidden_block_18(x))
        x = F.relu(self.hidden_block_19(x))
        output = self.end_block(x)
        return output


