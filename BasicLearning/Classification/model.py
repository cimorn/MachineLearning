import torch
import torch.nn as nn

class FashionNN(nn.Module):
    """Fashion-MNIST分类神经网络"""
    def __init__(self):
        super(FashionNN, self).__init__()
        self.flatten = nn.Flatten()  # 展平图像 (1x28x28 -> 784)
        self.layers = nn.Sequential(
            nn.Linear(784, 128),    # 输入层 -> 隐藏层1
            nn.ReLU(),              # 激活函数
            nn.Linear(128, 64),     # 隐藏层1 -> 隐藏层2
            nn.ReLU(),              # 激活函数
            nn.Linear(64, 10)       # 隐藏层2 -> 输出层（10个类别）
        )

    def forward(self, x):
        """前向传播"""
        x = self.flatten(x)
        logits = self.layers(x)  # 输出未归一化的概率
        return logits

if __name__ == "__main__":
    # 测试模型结构
    model = FashionNN()
    test_input = torch.randn(1, 1, 28, 28)  # 模拟单张图像
    output = model(test_input)
    print(f"模型输出形状: {output.shape}")  # 应输出 (1, 10)