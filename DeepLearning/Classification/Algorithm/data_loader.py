import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义Fashion-MNIST数据集的标签名称映射
# RGB是三个通道，这里是灰度图像所以只有一个通道
# 对应索引0-9分别代表的服装类别
label_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

def get_transform():
    """定义数据转换，让数据分布更接近标准正态分布。"""
    return transforms.Compose([
        transforms.ToTensor(),               # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1] # 这里会将像素值从[0,1]转换到[-1,1]范围
    ])

def load_data(batch_size=64):
    """加载Fashion-MNIST数据集"""
    # 获取定义好的数据转换操作
    transform = get_transform()
    
    # 加载训练集
    train_dataset = datasets.FashionMNIST(
        root='./data',         # 数据集存储路径
        train=True,            # 表示加载训练集
        download=True,         # 如果本地没有数据集则自动下载
        transform=transform    # 应用定义好的数据转换
    )
    
    # 加载测试集
    test_dataset = datasets.FashionMNIST(
        root='./data',         # 数据集存储路径
        train=False,           # 表示加载测试集
        download=True,         # 如果本地没有数据集则自动下载
        transform=transform    # 应用定义好的数据转换(与训练集保持一致)
    )
    
    # 创建训练集数据加载器
    train_loader = DataLoader(
        train_dataset,         # 要加载的训练数据集
        batch_size=batch_size, # 批次大小
        shuffle=True           # 训练时打乱数据顺序，增加随机性
    )
    
    # 创建测试集数据加载器
    test_loader = DataLoader(
        test_dataset,          # 要加载的测试数据集
        batch_size=batch_size, # 批次大小
        shuffle=False          # 测试时不需要打乱数据顺序
    )
    
    # 打印数据集基本信息
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    # 返回数据加载器和标签名称
    return train_loader, test_loader, label_names