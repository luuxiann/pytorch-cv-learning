## 图像增广
图像增广在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而扩大了训练集的规模。 此外，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力。
#### 基本图像增广方法
```
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()                       # 设置绘图的尺寸（d2l封装的matplotlib配置）
img = d2l.Image.open('./people.jpg')    # 打开图片（d2l.Image实际调用PIL库的Image）
d2l.plt.imshow(img)                     # 用matplotlib显示图像（d2l.plt即matplotlib.pyplot的别名）
d2l.plt.show()                          # 因为虚拟机图形显示配置问题，要显示调用才能显示图像

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    # 对原始图像img应用`aug`变换，生成 num_rows*num_cols 张增强后的图像,图像显示的缩放比例为scale
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    # 用d2l的工具函数显示所有增强图像，排列成 num_rows 行、num_cols 列
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# ---------------------- 翻转和裁剪 ----------------------
# 使用transforms模块来创建RandomFlipLeftRight实例，这样就各有50%的几率使图像向左或向右翻转。
apply(img, torchvision.transforms.RandomHorizontalFlip()) 
d2l.plt.show()
# RandomFlipTopBottom实例，使图像各有50%的几率向上或向下翻转。
apply(img, torchvision.transforms.RandomVerticalFlip())
d2l.plt.show()

# 随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值。 然后，区域的宽度和高度都被缩放到200像素。
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200),         # 裁剪后图像的目标尺寸（高200，宽200）
    scale=(0.1, 1),     # 裁剪区域面积占原始图像面积的比例范围（10%~100%）
    ratio=(0.5, 2)      # 裁剪区域宽高比的范围（0.5~2，即可以是窄长或宽扁）
)
apply(img, shape_aug)
d2l.plt.show()


# ---------------------- 改变颜色 ----------------------
# 随机更改图像的亮度，随机值为原始图像的50%（1 - 0.5）到150%（1 + 0.5）之间。
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
d2l.plt.show()

# 随机调整色调：色调偏移范围为-0.5~0.5（hue的取值范围通常为[-0.5, 0.5]）
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
d2l.plt.show()

# RandomColorJitter实例，同时随机更改图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
d2l.plt.show()


# Compose实例来综合上面定义的不同的图像增广方法，并将它们应用到每个图像。
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
d2l.plt.show()
```

|![computer_vision](./pictures/c1.png)|![computer_vision](./pictures/c2.png)|
|---|---|
|原图|左右翻转|
|![computer_vision](./pictures/c3.png)|![computer_vision](./pictures/c4.png)|
|上下翻转|裁剪|
|![computer_vision](./pictures/c5.png)|![computer_vision](./pictures/c6.png)|
|调亮度|调色调|
|![computer_vision](./pictures/c7.png)|![computer_vision](./pictures/c8.png)|
|亮度、对比度、饱和度和色调|综合|

#### 使用图像增广来训练模型
```
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 下载并加载CIFAR-10训练集  download=True：本地无数据时自动下载
all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
# 显示前32张图片，排列为4行8列，缩放比例0.8
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
d2l.plt.show()              ## 因为虚拟机图形显示配置问题，要显示调用才能显示图像

# 只使用最简单的随机左右翻转
# 使用ToTensor实例将一批图像转换为深度学习框架所要求的格式
# 即形状为（批量大小，通道数，高度，宽度）的32位浮点数，取值范围为0～1。
# 训练集增广：随机水平翻转 + 转为Tensor
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])
     
# 测试集无增广：仅转为Tensor（评估真实性能，不引入随机变换）
test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

# 将原始数据集转换为模型可直接训练的 “批量数据迭代器”，整合了数据读取、变换、批量处理逻辑。
def load_cifar10(is_train, augs, batch_size):
    # 加载CIFAR-10数据集，根据is_train区分训练/测试集
    dataset = torchvision.datasets.CIFAR10(
        root="../data", 
        train=is_train,  # True=训练集，False=测试集
        transform=augs,  # 应用定义的图像变换（增广+转Tensor）
        download=True
    )
    # 构建DataLoader：批量读取数据，支持打乱和多进程加速
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # 单次加载的样本数（已优化为64适配虚拟机）
        shuffle=is_train,       # 训练集打乱顺序（避免模型记忆），测试集不打乱
        num_workers=d2l.get_dataloader_workers()  # 自动获取适配的进程数（d2l封装）
    )
    return dataloader


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    # 将数据转移到第一个GPU（多GPU时DataParallel自动分发数据）
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]  # 适配列表型输入（如BERT微调）
    else:
        X = X.to(devices[0])                # 图像输入为张量，直接转移
    y = y.to(devices[0])                    # 标签转移到GPU

    net.train()          # 模型设为训练模式（启用Dropout、BN层更新）
    trainer.zero_grad()  # 清空上一批次的梯度（避免梯度累积）
    pred = net(X)        # 前向传播：输入图像→模型输出预测结果
    l = loss(pred, y)    # 计算每个样本的损失（reduction="none"保留单样本损失）
    l.sum().backward()   # 损失求和后反向传播（计算梯度）
    trainer.step()       # 优化器更新模型参数（用梯度调整权重）

    # 返回当前批次的总损失和总正确样本数
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)  # d2l封装的准确率计算（预测值vs真实标签）
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    """多GPU完整训练流程（多轮迭代+评估）"""
    timer, num_batches = d2l.Timer(), len(train_iter)  # 计时器（统计训练速度）、批次总数
    # 动画绘制器：实时展示训练损失、训练准确率、测试准确率
    animator = d2l.Animator(
        xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
        legend=['train loss', 'train acc', 'test acc']
    )
    # 多GPU封装：将模型复制到所有可用GPU，自动分发数据和聚合结果
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):  # 迭代num_epochs轮（此处为10轮）
        # 累加器：存储[总损失, 总正确数, 总样本数, 总标签数]
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):  # 迭代每个训练批次
            timer.start()
            # 训练当前批次，获取损失和准确率
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())  # 更新累加器
            timer.stop()

            # 每迭代1/5批次或最后一批次，记录中间结果（实时绘图）
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(
                    epoch + (i + 1) / num_batches,  # x轴：epoch + 批次比例（平滑展示）
                    (metric[0]/metric[2], metric[1]/metric[3], None)  # 训练损失、准确率
                )
        # 每轮训练结束后，在测试集评估准确率
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))  # 记录测试准确率

    # 训练结束，打印最终指标
    print(f'loss {metric[0]/metric[2]:.3f}, train acc {metric[1]/metric[3]:.3f}, test acc {test_acc:.3f}')
    # 打印训练速度（样本/秒）和使用的GPU
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

# 模型权重初始化：Xavier均匀分布（适合ReLU激活函数，加速收敛）
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:  # 仅对线性层和卷积层初始化
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)  # 递归应用初始化函数到所有子模块

# 带数据增广的训练函数（整合数据加载、损失函数、优化器）
def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    # 加载训练集和测试集（应用对应的增广策略）
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    # 损失函数：交叉熵损失（分类任务标准损失），reduction="none"保留单样本损失
    loss = nn.CrossEntropyLoss(reduction="none")
    # 优化器：Adam（自适应学习率，训练稳定，适合深度学习任务）
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    # 启动训练：10轮迭代，使用多GPU
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

# 执行训练（入口函数）
train_with_data_aug(train_augs, test_augs, net)
# 注释绘图显示：避免虚拟机图形配置问题和内存占用
# d2l.plt.show()
```
![computer_vision](./pictures/cc1.png)
![computer_vision](./pictures/c9.png)
也是跑完了，昨晚不知道为什么还给电脑跑重启了。

## 微调（fine-tuning）

微调是一种迁移学习（transfer learning）技术，其核心思想是将在大规模数据集上预训练好的模型参数迁移到新的任务中，通过在新任务的数据集上进行少量训练来调整这些参数，从而快速实现新任务的模型训练并获得较好的性能。这是因为预训练模型已经学习到了通用的特征提取能力，这些能力在许多相关任务中是可复用的。

#### 步骤
1. 在源数据集（例如ImageNet数据集）上预训练神经网络模型，即源模型。
2. 创建一个新的神经网络模型，即目标模型。这将复制源模型上的所有模型设计及其参数（输出层除外）。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。
3. 向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。
4. 在目标数据集（如椅子数据集）上训练目标模型。输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。

#### 热狗识别实例

```
import torch
from torch import nn
from torchvision import transforms, datasets
from d2l import torch as d2l

# 定义图像增广方法，用于数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小为224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # 调整大小为256
        transforms.CenterCrop(224),  # 中心裁剪为224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
data_dir = './hotdog'  # 数据集路径
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 获取设备（GPU或CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet-18模型
model_ft = torchvision.models.resnet18(pretrained=True)
# 获取最后一层的输入特征数
num_ftrs = model_ft.fc.in_features
# 修改输出层，适应二分类任务（热狗和非热狗）
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 学习率调度器，每7个epoch将学习率乘以0.1
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                # 训练时保留梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 训练阶段进行反向传播和参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 训练模型，迭代25个epoch
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
```

在上述代码中，首先定义了适用于训练集和验证集的图像增广方法，以增强模型的泛化能力。然后加载了预训练的ResNet-18模型，并根据热狗识别的二分类任务修改了输出层。在训练过程中，使用了交叉熵损失函数和SGD优化器，并通过学习率调度器调整学习率。通过训练，模型能够利用预训练得到的特征提取能力，快速适应热狗识别任务。

#### 冻结部分层训练与微调对比
- **冻结特征提取层**：只训练新修改的输出层，此时模型的特征提取部分使用预训练的参数，不进行更新。这种方式训练速度快，但可能无法充分适应新任务。
- **微调**：解冻部分或全部预训练层，让它们与输出层一起训练，只是这些层的学习率可能设置得较小。这种方式可以使模型更好地适应新任务，但训练时间更长，需要更多的计算资源。

通过对比可以发现，在数据量有限的情况下，微调通常能取得比仅训练输出层更好的性能，因为它允许模型根据新任务调整特征提取方式。

## torchvision

```
# Image Classification
import torch
from torchvision.transforms import v2

H, W = 32, 32
img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transforms(img)
```






