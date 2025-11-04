import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l

def f(x):
    """
    定义二次函数 f(x) = 3x² - 4x
    这是一个凸函数，用于演示导数和切线的概念
    """
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    """
    使用数值方法计算函数在给定点的导数
    采用前向差分公式：f'(x) ≈ [f(x+h) - f(x)] / h
    参数:
        f: 目标函数
        x: 计算导数的点
        h: 步长（越小越精确，但数值误差可能增大）
    返回:
        数值近似的导数值
    """
    return (f(x + h) - f(x)) / h

# 数值导数计算演示
# 通过不同步长展示数值导数的收敛性
h = 0.1  # 初始步长
print("数值导数计算 (在 x=1 处):")
print("步长越小，数值导数越接近理论值 2")
for i in range(5):
    # 计算在x=1处的数值导数
    derivative = numerical_lim(f, 1, h)
    print(f'h={h:.5f}, numerical limit={derivative:.5f}')
    h *= 0.1  # 每次迭代将步长缩小10倍

# 理论验证：
# f(x) = 3x² - 4x
# f'(x) = 6x - 4
# 在x=1处：f'(1) = 6*1 - 4 = 2
print(f"理论导数在 x=1 处: 2.00000")

def use_svg_display():  #@save
    """
    使用SVG格式显示绘图（在Jupyter Notebook中）
    SVG格式提供矢量图形，缩放不失真
    在普通Python脚本中此函数为空实现
    """
    pass

def set_figsize(figsize=(3.5, 2.5)):  #@save
    """
    设置matplotlib图表的大小
    参数:
        figsize: 元组 (宽度, 高度)，单位为英寸
    """
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    配置matplotlib坐标轴的各项属性
    参数:
        axes: matplotlib坐标轴对象
        xlabel, ylabel: x轴和y轴的标签文本
        xlim, ylim: x轴和y轴的显示范围 (min, max)
        xscale, yscale: 坐标轴尺度 ('linear'线性, 'log'对数)
        legend: 图例文本列表
    """
    # 设置坐标轴标签
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    
    # 设置坐标轴尺度
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    
    # 设置坐标轴范围
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    
    # 添加图例（如果提供）
    if legend:
        axes.legend(legend)
    
    # 添加网格线，提高可读性
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """
    通用绘图函数，用于绘制单条或多条数据曲线
    参数:
        X: x轴数据，可以是数值数组或列表
        Y: y轴数据，如果为None则绘制X自身
        xlabel, ylabel: 坐标轴标签
        legend: 图例说明列表
        xlim, ylim: 坐标轴范围限制
        xscale, yscale: 坐标轴尺度类型
        fmts: 线条格式列表，控制线条样式和颜色
        figsize: 图形尺寸
        axes: 可选的坐标轴对象
    """
    # 初始化图例列表
    if legend is None:
        legend = []

    # 设置图形尺寸
    set_figsize(figsize)
    
    # 获取或创建坐标轴对象
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        """
        判断输入数据是否是单维数据
        返回True如果:
          - X是1维numpy数组
          - X是标量列表（非嵌套列表）
        """
        return (hasattr(X, "ndim") and X.ndim == 1 or 
                isinstance(X, list) and not hasattr(X[0], "__len__"))

    # 数据预处理：统一数据格式
    # 确保X是列表形式，每个元素代表一条曲线的x数据
    if has_one_axis(X):
        X = [X]
    
    # 处理Y为None的情况（绘制X自身）
    if Y is None:
        X, Y = [[]] * len(X), X
    # 确保Y是列表形式
    elif has_one_axis(Y):
        Y = [Y]
    
    # 确保X和Y列表长度匹配
    if len(X) != len(Y):
        X = X * len(Y)
    
    # 清空当前坐标轴，准备绘制新图形
    axes.cla()
    
    # 遍历所有数据对，绘制每条曲线
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            # 绘制x-y数据对
            axes.plot(x, y, fmt)
        else:
            # 仅绘制y数据（x为索引）
            axes.plot(y, fmt)
    
    # 配置坐标轴属性
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 主程序：绘制函数和切线
# 生成x轴数据点，从0到3，步长0.1
x = np.arange(0, 3, 0.1)

# 绘制函数曲线和在x=1处的切线
# 函数：f(x) = 3x² - 4x
# 在x=1处的切线：y = f'(1)(x-1) + f(1) = 2(x-1) + (-1) = 2x - 3
plot(x, 
     [f(x), 2 * x - 3],  # 绘制函数曲线和切线
     'x',                 # x轴标签
     'f(x)',              # y轴标签
     legend=['f(x)', 'Tangent line (x=1)'],  # 图例说明
     xlim=[0, 3],         # x轴显示范围
     ylim=[-2, 10]        # y轴显示范围
)

# 显示图形（在Python脚本中必需）
plt.show()