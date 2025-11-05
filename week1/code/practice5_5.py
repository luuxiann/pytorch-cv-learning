import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l

def f(x):
    return np.sin(x)

# 替代原前向差分（更精确），且不依赖cos(x)解析解
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)  # 中心差分公式，逼近f'(x)

# 数值验证
h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical_derivative={numerical_derivative(f, 1, h):.5f}')
    h *= 0.1

# 绘图辅助函数
def use_svg_display():
    pass

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

x = np.linspace(-np.pi, np.pi, 100)   # x范围为-π~π（覆盖sin(x)一个周期）
y_f = f(x)                            # 计算f(x)=sin(x)的函数值
y_deriv = numerical_derivative(f, x)  # 计算f(x)的数值导数（不用cos(x)）

# 绘制图像
plot(x, [y_f, y_deriv],                                      # Y轴数据：[函数值, 数值导数]
     xlabel='x', ylabel='y',                                 # 坐标轴标签
     legend=['$f(x)=\sin(x)$', '$f\'(x)'],
     xlim=(-np.pi, np.pi), ylim=(-1.2, 1.2),                 # 适配sin(x)的值域（-1~1）
     figsize=(8, 4))                                         # 放大图像尺寸，更清晰
plt.title('$\sin(x)$  and derivative')  # 标题
plt.show()