import torch.nn as nn
import torch.optim as optim
import random
import torch

def train_model(model, train_loader, epochs=100, lr=0.03):
    """训练模型"""
    # 定义损失函数和优化器（应在循环外定义，避免重复初始化）
    criterion = nn.MSELoss()  # 定义均方误差损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr)  # 随机梯度下降优化器
    
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





def verify(model1, model2, true_w, true_b, num_verify=5, seed=42):
    """验证两个模型的预测值：对比理论值、带噪声标签及两个模型的预测结果"""
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 随机生成全新特征（与训练数据同分布）
    input_size = len(true_w)
    sample_X = torch.normal(0, 1, (num_verify, input_size))  # 全新随机特征
    
    # 计算理论值和带噪声标签
    #sample_y_true = torch.matmul(sample_X, true_w) + true_b  # 无噪声理论值
    sample_y_true = torch.matmul( torch.sin(sample_X)*sample_X, true_w) + true_b  # 无噪声理论值

    sample_y_noisy = sample_y_true + torch.normal(0, 0.01, sample_y_true.shape)  # 带噪声标签
    
    # 两个模型分别预测
    model1.eval()
    model2.eval()
    with torch.no_grad():
        pred1 = model1(sample_X)  # 模型1预测值
        pred2 = model2(sample_X)  # 模型2预测值
    
    # 定义列宽（第一列加宽以显示特征值）
    widths = [50, 15, 15, 15, 15, 15, 15]  # 第一列宽度增加到30（容纳特征值列表）
    headers = ["测试值", "理论值(无噪声)", "带噪声标签", "Linear模型预测值", "Linear模型误差", "Deep模型预测值", "Deep模型误差"]
    
    # 打印标题
    print("\n===== 两个模型验证结果对比 =====")
    header_line = ""
    for i in range(len(headers)):
        header_line += headers[i].ljust(widths[i]) + "|"
    print(header_line)
    
    # 打印分隔线
    sep_line = ""
    for w in widths:
        sep_line += "-" * w + "-----"
    sep_line += "----" 
    print(sep_line)
    
    # 打印数据行
    for i in range(num_verify):
        # 第一列：测试特征值（格式化显示为列表）
        x_vals = [f"{x:.4f}" for x in sample_X[i].numpy()]
        test_val_str = "[" + ", ".join(x_vals) + "]"  # 例如：[0.3452, -1.2345]
        
        # 其他数据
        true_val = sample_y_true[i].item()
        noisy_val = sample_y_noisy[i].item()
        p1 = pred1[i].item()
        p2 = pred2[i].item()
        e1 = abs(p1 - noisy_val)
        e2 = abs(p2 - noisy_val)
        
        # 格式化行（左对齐）
        row = (
            f"{test_val_str:<{widths[0]+3}}|"  # 第一列：测试值
            f"{true_val:<{widths[1]+6}.4f}|"   # 理论值
            f"{noisy_val:<{widths[2]+5}.4f}|"  # 带噪声标签
            f"{p1:<{widths[3]+5}.6f}|"         # 模型1预测值
            f"{e1:<{widths[4]+4}.6f}|"         # 模型1误差
            f"{p2:<{widths[5]+5}.6f}|"         # 模型2预测值
            f"{e2:<{widths[6]+4}.6f}|"         # 模型2误差
        )
        print(row)