import pandas as pd
import torch
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

# 加载数据并划分训练集(80%)和测试集(20%)
df = pd.read_csv('house_rent_dataset.csv')
X = df[['area', 'bedrooms', 'subway_dist']].values
y = df['rent'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 固定随机种子，划分结果一致
)

# 转换为PyTorch张量
train_features = torch.tensor(X_train, dtype=torch.float32)
train_labels = torch.tensor(y_train, dtype=torch.float32)
test_features = torch.tensor(X_test, dtype=torch.float32)
test_labels = torch.tensor(y_test, dtype=torch.float32)


# 数据加载
def load_array(arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 32
train_iter = load_array((train_features, train_labels), batch_size)
test_iter = load_array((test_features, test_labels), batch_size, is_train=False)

# 定义模型
net = nn.Sequential(nn.Linear(3, 1))
net[0].weight.data.normal_(0, 0.001)
net[0].bias.data.fill_(0)

# 损失函数和优化算法
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.00001)

# 训练模型
num_epochs = 25
train_losses = []   # 存储所有训练集损失值

print("训练模型")
for epoch in range(num_epochs):
    train_total = 0.0
    net.train()  # 训练模式
    for X, y in train_iter:
        y_pred = net(X)
        l = loss(y_pred, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        train_total += l.item() * X.shape[0]
    train_avg = train_total / len(train_features)
    train_losses.append(train_avg)

    # 每5轮打印训练进度
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch + 1}/{num_epochs}，训练损失：{train_avg:.2f}")

print("\n测试模型")

# 测试模型
net.eval()
test_total = 0.0
all_preds = []  # 存储所有测试集预测值
all_trues = []  # 存储所有测试集真实值

with torch.no_grad():  # 禁用梯度计算，节省资源
    for X, y in test_iter:
        y_pred = net(X)
        l = loss(y_pred, y)
        test_total += l.item() * X.shape[0]
        # 收集所有预测值和真实值
        all_preds.extend(y_pred.numpy().flatten())
        all_trues.extend(y.numpy().flatten())

test_avg = test_total / len(test_features)
print(f"测试集平均损失：{test_avg:.2f}")

# 可视化训练损失曲线
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 输出模型参数
w = net[0].weight.data.numpy()[0]
b = net[0].bias.data.item()
print("\n模型参数：")
print(f"面积权重：{w[0]:.2f} 元/㎡")
print(f"卧室数权重：{w[1]:.2f} 元/个")
print(f"地铁距离权重：{w[2]:.2f} 元/km")
print(f"基准租金：{b:.2f} 元/月")

# 测试集随机抽10个样本验证
sample_indices = random.sample(range(len(all_trues)), 10)  # 随机选10个测试样本索引
print("\n测试集样本预测（10个随机样本）：")
for i, idx in enumerate(sample_indices, 1):
    true_val = all_trues[idx]
    pred_val = all_preds[idx]
    print(f"样本{i}：真实{true_val:.0f}元 → 预测{pred_val:.0f}元，误差{abs(pred_val - true_val):.0f}元")