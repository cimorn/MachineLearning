import torch  # 导入PyTorch框架
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import matplotlib.pyplot as plt  # 导入绘图库，用于可视化
import os  # 导入操作系统接口模块


# 确保results文件夹存在，不存在则创建
os.makedirs('results', exist_ok=True)

def train_model(model, train_loader, epochs=10, lr=0.001):
    # 定义损失函数：交叉熵损失（适用于多分类问题）
    # 包含了softmax激活和负对数似然损失的组合
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：Adam优化器（常用的自适应学习率优化器）
    # 传入模型参数和学习率
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model.train()  # 切换模型到训练模式（启用 dropout/batchnorm等训练特定层）
    train_losses = []  # 存储每轮的平均损失，用于后续可视化
    
    # 迭代训练指定轮数
    for epoch in range(epochs):
        total_loss = 0.0  # 累计当前轮的总损失
        
        # 遍历训练集中的每个批次
        for images, labels in train_loader:
            # 前向传播：将输入图像传入模型，得到预测输出
            outputs = model(images)
            # 计算当前批次的损失（预测输出与真实标签的差距）
            loss = criterion(outputs, labels)
            
            # 反向传播与参数优化
            optimizer.zero_grad()  # 清空上一轮的梯度（避免梯度累积）
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 根据梯度更新模型参数
            
            total_loss += loss.item()  # 累加当前批次的损失值（.item()获取标量值）
        
        # 计算当前轮的平均损失（总损失 / 批次数量）
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)  # 记录平均损失
        # 打印训练进度
        print(f'Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.4f}')
    
    return train_losses  # 返回训练过程中的损失变化

def test_model(model, test_loader):
    model.eval()  # 切换模型到评估模式（关闭 dropout/batchnorm等训练特定层）
    correct = 0   # 记录正确预测的样本数
    total = 0     # 记录总样本数
    
    # 关闭梯度计算（测试阶段不需要反向传播，节省内存和计算资源）
    with torch.no_grad():
        # 遍历测试集中的每个批次
        for images, labels in test_loader:
            # 前向传播获取预测结果
            outputs = model(images)
            # 取输出中概率最大的类别作为预测结果
            # torch.max返回最大值和对应索引，这里用_忽略最大值，保留索引（预测类别）
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)  # 累加总样本数（当前批次的样本数）
            # 累加正确预测的样本数（预测类别与真实标签一致的数量）
            correct += (predicted == labels).sum().item()
    
    # 计算准确率（正确数/总数 * 100转换为百分比）
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy

def plot_losses(losses):
    plt.figure(figsize=(10, 4))  # 创建绘图窗口，设置大小
    # 绘制损失曲线（x轴为轮数，y轴为损失值）
    plt.plot(range(1, len(losses)+1), losses)
    plt.title('Training Loss Curve')  # 设置图表标题
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.ylabel('Average Loss')  # 设置y轴标签
    plt.grid(alpha=0.3)  # 添加网格线，增强可读性
    
    # 保存图像到results文件夹，不显示
    plt.savefig('results/loss_curve.png', bbox_inches='tight')  # bbox_inches避免标签被截断
    plt.close()  # 关闭当前图像，释放内存

def visualize_predictions(model, test_loader, label_names, num_samples=10):
    model.eval()  # 切换到评估模式
    # 获取测试集中的一个批次数据（图像和对应标签）
    images, labels = next(iter(test_loader))
    # 对前num_samples个样本进行预测
    outputs = model(images[:num_samples])
    _, predicted = torch.max(outputs, 1)  # 获取预测类别
    
    # 创建绘图窗口
    plt.figure(figsize=(12, 6))
    # 循环绘制每个样本的预测结果
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)  # 创建2行5列的子图，定位到第i+1个位置
        
        # 反标准化图像（恢复到[0,1]范围便于显示）
        img = images[i].numpy().squeeze() * 0.5 + 0.5
        plt.imshow(img, cmap='gray')  # 以灰度图显示
        
        # 获取真实标签和预测标签的名称
        true_label = label_names[labels[i]]
        pred_label = label_names[predicted[i]]
        
        # 标签颜色：预测正确为绿色，错误为红色
        color = 'green' if true_label == pred_label else 'red'
        # 设置子图标题
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.axis('off')  # 关闭坐标轴显示
    
    plt.tight_layout()  # 自动调整子图间距
    # 保存图像到results文件夹，不显示
    plt.savefig('results/predictions.png', bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存