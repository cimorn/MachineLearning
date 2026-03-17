import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # 修正此处的导入错误

plt.rcParams["font.family"] = ["Heiti TC"]  # 支持中文的字体列表
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# Fashion-MNIST类别名称（0-9对应）
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def load_fashion_mnist(data_dir):
    """加载指定路径下的Fashion-MNIST数据集（适配torchvision默认路径）"""
    # 数据集文件相对路径（对应 data/FashionMNIST/raw/ 下的文件）
    files = {
        'train_images': 'train-images-idx3-ubyte',
        'train_labels': 'train-labels-idx1-ubyte',
        'test_images': 't10k-images-idx3-ubyte',
        'test_labels': 't10k-labels-idx1-ubyte'
    }
    
    # 检查文件是否存在
    for name, filename in files.items():
        file_path = os.path.join(data_dir, filename)
        # 同时检查是否有.gz压缩版本（有些下载的是压缩文件）
        if not os.path.exists(file_path):
            file_path_gz = f"{file_path}.gz"
            if os.path.exists(file_path_gz):
                file_path = file_path_gz  # 使用压缩文件
            else:
                raise FileNotFoundError(f"未找到文件：{file_path} 或 {file_path_gz}")
        files[name] = file_path  # 更新为实际存在的路径

    # 读取图像（二进制文件格式）
    def read_images(file_path):
        # 判断是否为压缩文件
        opener = gzip.open if file_path.endswith('.gz') else open
        with opener(file_path, 'rb') as f:
            # 跳过文件头（前16字节：魔数、样本数、行数、列数）
            f.read(4)  # 魔数
            num_images = int.from_bytes(f.read(4), byteorder='big')
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')
            # 读取像素数据并reshape
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows, cols)
    
    # 读取标签（二进制文件格式）
    def read_labels(file_path):
        opener = gzip.open if file_path.endswith('.gz') else open
        with opener(file_path, 'rb') as f:
            # 跳过文件头（前8字节：魔数、样本数）
            f.read(4)  # 魔数
            num_labels = int.from_bytes(f.read(4), byteorder='big')
            # 读取标签数据
            return np.frombuffer(f.read(), dtype=np.uint8)

    # 加载训练集和测试集
    train_images = read_images(files['train_images'])
    train_labels = read_labels(files['train_labels'])
    test_images = read_images(files['test_images'])
    test_labels = read_labels(files['test_labels'])
    
    return {
        'train': {'images': train_images, 'labels': train_labels},
        'test': {'images': test_images, 'labels': test_labels}
    }

def analyze_dataset(dataset):
    """分析数据集并可视化结果"""
    # 1. 基本信息
    print(f"训练集样本数：{len(dataset['train']['images'])}")
    print(f"测试集样本数：{len(dataset['test']['images'])}")
    print(f"图像尺寸：{dataset['train']['images'].shape[1:]} 像素（灰度图）\n")

    # 2. 类别分布
    def count_labels(labels):
        counts = defaultdict(int)
        for label in labels:
            counts[CLASS_NAMES[label]] += 1
        return counts
    
    train_counts = count_labels(dataset['train']['labels'])
    test_counts = count_labels(dataset['test']['labels'])
    
    print("训练集类别分布：")
    for cls, cnt in train_counts.items():
        print(f"  {cls}: {cnt}")
    print("\n测试集类别分布：")
    for cls, cnt in test_counts.items():
        print(f"  {cls}: {cnt}")

    # 3. 可视化每个类别的样本
    plt.figure(figsize=(12, 6))
    for i in range(10):
        # 找到第i类的第一个样本
        idx = np.where(dataset['train']['labels'] == i)[0][0]
        img = dataset['train']['images'][idx]
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(CLASS_NAMES[i], fontsize=10)
        plt.axis('off')
    plt.suptitle("每个类别的样本示例", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 4. 像素值分布（训练集）
    all_pixels = dataset['train']['images'].flatten()
    plt.figure(figsize=(10, 4))
    plt.hist(all_pixels, bins=50, color='gray', alpha=0.7)
    plt.title("训练集像素值分布（0-255）")
    plt.xlabel("像素值")
    plt.ylabel("频数")
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 数据集所在路径（根据你的路径修改，这里是 data/FashionMNIST/raw/）
    data_dir = "./data/FashionMNIST/raw"
    
    try:
        print("正在加载Fashion-MNIST数据集...")
        dataset = load_fashion_mnist(data_dir)
        print("数据集加载成功！开始分析...\n")
        analyze_dataset(dataset)
    except Exception as e:
        print(f"错误：{e}")