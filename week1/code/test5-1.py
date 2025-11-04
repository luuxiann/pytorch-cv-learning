import torch
# y=2 * x^T * x
x = torch.arange(4.0)
print(x)
# 为张量开启自动求导功能，并提示此时梯度尚未计算（为 None）
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True),requires_grad=True用于开启自动求导追踪
print(x.grad)  # 用于访问张量 x 的梯度值,默认值是None
print(x)

y = 2 * torch.dot(x, x)
print(y)

y.backward()		# 调用反向传播函数来自动计算y关于x每个分量的梯度
print(x.grad)			# 打印这些梯度