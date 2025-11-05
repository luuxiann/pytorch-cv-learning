import torch
def f(x):
    y = x
    
    while y < 5:    # y小于5时持续加1
        y = y + 1
    if y > 6:       # y大于6则乘2，否则乘3
        z = y * 2
    else:
        z = y * 3
    return z

x = torch.randn(size=(), requires_grad=True)  
z = f(x)
z.backward()
print(x.grad)
