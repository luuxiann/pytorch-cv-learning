import numpy as np
import torch
from torch.utils import data    # 用于数据加载的工具库
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn            # nn模块提供神经网络相关的层、损失函数等

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 读取数据集: 调用框架中现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):  #@save # is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
    """构造一个PyTorch数据迭代器"""
    """
        构造PyTorch数据迭代器，用于按批次加载数据
        参数：
            data_arrays: 包含特征和标签的元组（如(features, labels)）
            batch_size: 每个批次的样本数
            is_train: 是否在训练时打乱数据（True表示打乱，用于训练；False用于测试）
        返回：
            DataLoader对象：可迭代的批次数据生成器
        """
    # TensorDataset：将特征和标签打包成数据集（按索引对应）
    dataset = data.TensorDataset(*data_arrays)  # *用于解包元组，等价于dataset = TensorDataset(features, labels)
    # DataLoader：按批次加载数据集，支持打乱、多线程等
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

#将features和labels作为API的参数传递，并通过数据迭代器指定batch_size。
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 验证是否正常工作，读取并打印第一个小批量样本。
print(next(iter(data_iter)))

# 定义模型
'''
定义一个模型变量net，是一个Sequential类的实例。 
Sequential类将多个层串联在一起。 
当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。
该例子只包含一个层，因此实际上不需要Sequential。
'''
net = nn.Sequential(nn.Linear(2, 1)) # nn.Linear(2, 1)：输入特征数为2，输出特征数为1的线性层（即y = X·w + b）

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # net[0]表示Sequential中的第一个层
net[0].bias.data.fill_(0)           # 替换方法normal_和fill_重写参数值。

# 定义损失函数，MSELoss类，返回所有样本损失的平均值。
loss = nn.MSELoss()

# 定义优化函数
#   net.parameters()：需要优化的参数（自动获取net中所有requires_grad=True的参数，即w和b）
#   lr=0.03：学习率（步长）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# 训练
# 计算每个迭代周期后的损失，并打印来监控训练过程
# 训练轮数：遍历整个数据集3次
num_epochs = 3
for epoch in range(num_epochs):
    # 遍历所有批次数据
    for X, y in data_iter:
        # 1. 前向传播：计算当前批次的预测值并求损失
        l = loss(net(X), y)  # net(X)自动调用前向传播，输出预测值
        # 2. 梯度清零：避免当前批次的梯度与上一批次累积
        trainer.zero_grad()  # 等价于手动实现的param.grad.zero_()
        # 3. 反向传播：计算损失对参数的梯度
        l.backward()  # 自动计算w和b的梯度
        # 4. 参数更新：根据梯度和学习率更新参数
        trainer.step()  # 等价于手动实现的参数更新公式
    # 每轮结束后，计算全量数据的损失（评估训练效果）
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')  # 损失应逐渐减小，表明模型在优化

# 验证训练结果
# 比较生成数据集的真实参数和通过有限数据训练获得的模型参数
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)


