import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
    
a = torch.randn(3, requires_grad=True)  # 随机向量
print(a)
d = f(a)
d.sum().backward()  # 对向量的和求导（转为标量）
print(a.grad)