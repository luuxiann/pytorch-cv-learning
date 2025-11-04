# Pytorch学习
## 教程
李沐老师的《动手学深度学习》：https://www.bilibili.com/video/BV1f54ZzGEMC/?spm_id_from=333.337.search-card.all.click&vd_source=301feeee8ee482b3882c9f098e564491
### 资源
![problem1](./pictures/1.png)
https://github.com/d2l-ai
学习内容见：https://zh-v2.d2l.ai/ 
以课件学习为主，视频学习为辅。

### 知识点
#### 预备知识

1. 张量(tensor) ： 表示一个由数值组成的数组，这个数组可能有多个维度。张量中的每个值都称为张量的 元素（element）。新的张量将存储在内存中，并采用基于CPU的计算。
   深度学习存储和操作数据的主要接口
   需导入torch
    ```import torch```
    ```
    x = torch.arange(12)        # arange 创建一个行向量 x
    x
    > tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    x.shape                     # shape属性来访问张量（沿每个轴的长度）的形状 
    > torch.Size([12])

    x.numel()                   # 张量中元素的总数，即形状的所有元素乘积
    > 12

    X = x.reshape(3, 4)         # 改变一个张量的形状而不改变元素数量和元素值
    X      
    >  tensor([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])   
    # 不需要通过手动指定每个维度来改变形状。指定其中的n-1个维度，剩下一个维度可用-1代替，如上面的例子可用x.reshape(-1,4)或x.reshape(3,-1)
    torch.zeros((2, 3, 4))      # 其中所有元素都设置为0
    torch.ones((2, 3, 4))       # 其中所有元素都设置为0
    # （2，3，4）三维，可以看到有三个方括号
    > tensor([[[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]],

              [[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]]]) 
    torch.randn(3, 4)           # 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
    > tensor([[-0.0135,  0.0665,  0.0912,  0.3212],
              [ 1.4653,  0.1843, -1.6995, -0.3036],
              [ 1.7646,  1.0450,  0.2457, -0.7732]])
    
    #直接赋值
    torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    > tensor([[2, 1, 4, 3],
              [1, 2, 3, 4],
              [4, 3, 2, 1]])
    
    B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
    ```
    ```
    # 张量形状相同
    # 标准算术运算符
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
    > (tensor([ 3.,  4.,  6., 10.]),
       tensor([-1.,  0.,  2.,  6.]),
       tensor([ 2.,  4.,  8., 16.]),
       tensor([0.5000, 1.0000, 2.0000, 4.0000]),
       tensor([ 1.,  4., 16., 64.]))
    
    # 把多个张量连结（concatenate）在一起
    X = torch.arange(12, dtype=torch.float32).reshape((3,4)) # 浮点型，3 X 4
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    torch.cat((X, Y), dim=0)            #沿行
    torch.cat((X, Y), dim=1)            #沿列
    > (tensor([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.],
               [ 8.,  9., 10., 11.],
               [ 2.,  1.,  4.,  3.],
               [ 1.,  2.,  3.,  4.],
               [ 4.,  3.,  2.,  1.]]),
       tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
               [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
               [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))

    # 逻辑运算符
    # 对于每个位置，如果X和Y在该位置相等，则新张量中相应项的值为1。 这意味着逻辑语句X == Y在该位置处为真，否则该位置为0。
    X == Y                              # X和Y维度形状相同
    > tensor([[False,  True, False,  True],
              [False, False, False, False],
              [False, False, False, False]])

    X.sum()                             # 对张量中的所有元素进行求和
    > tensor(66.)
    ```
    
    广播机制（broadcasting mechanism）
    1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
    2. 对生成的数组执行按元素操作。
    ```
    # 张量形状不同
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    a, b
    a + b
    (tensor([[0],
             [1],
             [2]]),
     tensor([[0, 1]]))
     tensor([[0, 1],
             [1, 2],
             [2, 3]])
    # 将两个矩阵广播为一个更大的 3 X 2矩阵，矩阵a将复制列， 矩阵b将复制行，然后再按元素相加
    ```
    索引和切片
    索引访问张量中的元素
    ![pytorch_learning1](./pictures/2.png)
    节省内存
    ```
    # 使用切片表示法将操作的结果分配给先前分配的数组
    X[:] = <expression>
    X[:] = X + Y 或 X += Y
    ```

    转换为其他Python对象
    ```
    # NumPy张量（ndarray）转换为张量
    A = X.numpy()
    B = torch.tensor(A)
    type(A), type(B)
    > (numpy.ndarray, torch.Tensor)

    # 大小为1的张量转换为Python标量
    a = torch.tensor([3.5])
    a, a.item(), float(a), int(a)
    > (tensor([3.5000]), 3.5, 3.5, 3)
    ```
    练习
    1. 将本节中的条件语句X == Y更改为X < Y或X > Y，然后看看你可以得到什么样的张量。
    2. 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？
    **广播机制中的两个张量，要能通过复制整个行或列拓展为形状相同的张量。如形状为3 X 4和4 X 3的两个张量就不能通过广播机制来进行运算。** 
    代码见code/practice1
    结果：
    ![pytorch_learning1](./pictures/3.png)
2. 数据预处理
   使用pandas预处理原始数据，并将原始数据转换为张量格式的步骤。
   1. 读取数据集
        ```
        # 导入pandas包并调用read_csv函数
        import os
        import pandas as pd

        os.makedirs(os.path.join('..', 'data'), exist_ok=True)
        data_file = os.path.join('..', 'data', 'house_tiny.csv')
        with open(data_file, 'w') as f:
            f.write('NumRooms,Alley,Price\n')  # 列名
            f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
            f.write('2,NA,106000\n')
            f.write('4,NA,178100\n')
            f.write('NA,NA,140000\n')
        data = pd.read_csv(data_file)
        print(data)
        >    NumRooms Alley   Price
         0       NaN  Pave  127500
         1       2.0   NaN  106000
         2       4.0   NaN  178100
         3       NaN   NaN  140000
        ```
    1. 处理缺失值 NaN
        插值法用一个替代值弥补缺失值
        ```
        inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
        # 处理 NumRooms离散值，用均值填充
        inputs = inputs.fillna(inputs.mean())               
        # 处理Alley类型值，转换成两列
        inputs = pd.get_dummies(inputs, dummy_na=True)
        print(inputs)
        >         NumRooms  Alley_Pave  Alley_nan
            0       3.0           1          0
            1       2.0           0          1
            2       4.0           0          1
            3       3.0           0          1
        ```
        运行inputs = inputs.fillna(inputs.mean())时，因无法处理Alley类型，会报错
            ![pytorch_learning](./pictures/4.png)

    3. 转换为张量格式
        ```
        import torch

        x = torch.tensor(inputs.to_numpy(dtype=float))
        y = torch.tensor(outputs.to_numpy(dtype=float))
        print(x)
        print(y)
        (tensor([[3., 1., 0.],
                 [2., 0., 1.],
                 [4., 0., 1.],
                 [3., 0., 1.]], dtype=torch.float64),
         tensor([127500., 106000., 178100., 140000.], dtype=torch.float64))
         ```
    4. 练习
        创建包含更多行和列的原始数据集。
       1. 删除缺失值最多的列。
       2. 将预处理后的数据集转换为张量格式。
       代码见code/practice2
        结果：
        ![pytorch_learning](./pictures/5.png)
3. 线性代数
   1. 标量（scalar）、向量
   2. 长度、维度和形状
        ```
        len(x)      # 长度
                    # 维度看有几行
        x.shape     # 形状
        ```
    3. 矩阵
        ```
        A.T             # 矩阵的转置
        ```
    4. 张量算法的基本性质
       1. 两个相同形状的张量相加或相乘，对应元素相加或相乘（Hadamard积）
       2. 张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
    5. 降维
        ```
        A_sum_axis0 = A.sum(axis=0)     # 通过求和所有行的元素来降维（轴0）
        A_sum_axis1 = A.sum(axis=1)     # 通过求和所有列的元素来降维（轴1）
        A.sum(axis=[0, 1])或A.sum()     # 求和
        A.mean()或 A.sum() / A.numel()  # 求平均值
        # 沿指定轴降低张量的维度
        A.mean(axis=0)或A.sum(axis=0) / A.shape[0]
        ```
    6. 不降维
        ```
        sum_A = A.sum(axis=1, keepdims=True)    # 计算总和或均值时保持轴数不变
        A.cumsum(axis=0)                        # 沿某个轴计算A元素的累积总和,不会沿任何轴降低输入张量的维度。
        ```
    7. 乘
        ```
        torch.dot(A,B) 或 torch.sum(A * B)     # 点积（Dot Product）
        torch.mv(A, B)                          # 向量积，A的列维数（沿轴1的长度）必须与B的维数（其长度）相同
        torch.mm(A, B)                          # 矩阵-矩阵乘法（matrix-matrix multiplication
        ``` 
    8. 范数（norm）
        向量的范数是表示一个向量有多大。
        |![pytorch_learning](./pictures/6.png)|![pytorch_learning](./pictures/7.png)|
        |--|--|
        |![pytorch_learning](./pictures/9.png)|![pytorch_learning](./pictures/8.png)|

        目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。
    9. 练习
        ![pytorch_learning](./pictures/12.png)
        代码见code/practice3
        结果：
        |![pytorch_learning](./pictures/10.png)|![pytorch_learning](./pictures/11.png)|
        |--|--|