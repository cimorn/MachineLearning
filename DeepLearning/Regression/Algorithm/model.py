import torch.nn as nn

class LinearRegressionModel(nn.Module):
    """线性回归模型"""
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层

    def forward(self, x):
        """前向传播"""
        return self.linear(x)



class DeepRegressionModel(nn.Module):
    """深度神经网络回归模型"""
    def __init__(self, input_size):
        super(DeepRegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),   # 隐藏层1
            nn.ReLU(),                   # 激活函数引入非线性
            nn.Linear(64, 32),           # 隐藏层2
            nn.ReLU(),
            nn.Linear(32, 1)             # 输出层
        )

    def forward(self, x):
        """前向传播"""
        return self.layers(x)
