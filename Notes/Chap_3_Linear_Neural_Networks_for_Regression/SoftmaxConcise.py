import torch
from torch import nn
from d2l import torch as d2l

def init_weights(m):    # 初始化模型参数
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))   # 定义模型，Flatten层将输入的形状展平成单个维度，然后输入到Linear层
net.apply(init_weights)

# softmax只是一种求概率的方法
# softmax和交叉熵损失函数结合在了一起，这样可以避免数值不稳定
# 因此上面并没有提到softmax函数，因为在交叉熵损失函数中已经包含了softmax运算
loss = nn.CrossEntropyLoss()   # 定义损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
