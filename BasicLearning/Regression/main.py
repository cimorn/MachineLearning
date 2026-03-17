import random
import torch
import torch.nn as nn
import torch.optim as optim
from data import synthetic_data, get_data_loader
from model import LinearRegressionModel, DeepRegressionModel
from plotter import plot_scatter, plot_loss
from trainer import train_model, verify


#可以根据数据生成的方式不同和验证模型的数据方式不同进行修改
# data.py文件 第8行
# trainer.py文件 第55行

def train_model(model, train_loader, epochs=100, lr=0.03):
    """训练模型"""
    # 定义损失函数和优化器（应在循环外定义，避免重复初始化）
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr)  # 随机梯度下降优化器
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()  # 切换到训练模式
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
                       
        for X, y in train_loader:
            # 前向传播：计算预测值
            y_pred = model(X)
            
            # 计算损失（使用实例化后的损失函数）
            loss = criterion(y_pred, y)

            # 反向传播与优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新参数
            
            total_loss += loss.item()
        
        # 记录损失（通常取平均损失更合理）
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        # 每10个epoch打印一次训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}')
    
    return losses


def main():
    # 设置随机种子，保证结果可复现
    seed=42
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 1. 设置参数
    true_w = torch.tensor([2.0, -3.4])  # 真实权重
    true_b = 4.2 # 真实偏差
    num_samples = 1000  # 样本数量
    epochs = 100 #训练轮数
    lr = 0.01  #model学习率
    ver_num=10  # 验证样本数量


    # 2. 生成数据
    features, labels = synthetic_data(true_w, true_b, num_samples)
    print(f"\n特征矩阵形状: {features.shape}, 标签向量形状: {labels.shape}, 真实权重: {true_w}, 真实偏差: {true_b}\n")
    plot_scatter(features, labels, save_path='results/data')
    
    # 3. 创建数据加载器
    train_loader = get_data_loader(features, labels, batch_size=32)
    
    # 4. 初始化模型（可以切换为DeepRegressionModel）
    input_size = len(true_w)  # 输入特征数

    # 5. 训练线性模型
    print("\n开始训练线性模型...")
    model_linear =LinearRegressionModel(input_size)
    losses_linear = train_model(model_linear, train_loader, epochs=epochs, lr=lr)
    plot_loss(losses_linear, save_path='results/loss_linear')  # 传入训练过程中记录的损失列表

    # 5. 训练深度模型
    print("\n开始训练线性模型...")
    model_deep =DeepRegressionModel(input_size)  # 可选的深度模型
    losses_deep = train_model(model_deep, train_loader, epochs=epochs, lr=lr)
    plot_loss(losses_deep, save_path='results/loss_deep')  # 传入训练过程中记录的损失列表


    verify(model_linear, model_deep, true_w, true_b, num_verify=ver_num, seed=seed)  # 验证5个样本

if __name__ == "__main__":

    main()