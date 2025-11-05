import torch
# y=2 * x^T * x
x = torch.arange(4.0)
# 为张量开启自动求导功能，并提示此时梯度尚未计算（为 None）
x.requires_grad_(True)
y = 2 * torch.dot(x, x)

y.backward()		# 调用反向传播函数来自动计算y关于x每个分量的梯度
print(x.grad)		# 打印这些梯度
y.backward()		# 再次调用反向传播函数
print(x.grad)