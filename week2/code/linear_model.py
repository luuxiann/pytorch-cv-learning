# 导入库
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn
import matplotlib.pyplot as plt

# 生成模拟数据：y = 2*x1 -3.4*x2 +4.2 + 噪声
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 可视化
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)
plt.xlabel('Feature 2')
plt.ylabel('Label')
plt.show()

# 加载数据
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print("第一批次数据：", next(iter(data_iter)))  # 验证数据

# 定义线性回归模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 计算损失
        trainer.zero_grad()  # 清零梯度
        l.backward()         # 反向传播
        trainer.step()       # 更新参数
    l = loss(net(features), labels)  # 计算全量损失
    print(f'epoch {epoch + 1}, loss {l:f}')

# 输出参数误差
w = net[0].weight.data
b = net[0].bias.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)

# 参数对比
print("\n=== 参数对比 ===")
print(f"真实权重：{true_w}，学习权重：{w.reshape(true_w.shape)}")
print(f"真实偏置：{true_b}，学习偏置：{b.item()}")
