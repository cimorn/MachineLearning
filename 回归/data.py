import torch

def synthetic_data(w, b, num_examples):
    """生成y = Xw + b + 噪声"""
    # 生成均值为0、标准差为1的随机特征矩阵
    X = torch.normal(0, 1, (num_examples, len(w)))

    y = torch.matmul( X, w) + b
    # 计算线性模型结果 y = Xw + b
    # y = torch.matmul( torch.sin(X)*X, w) + b
    # # 计算线性模型结果 y = sinx*x + b   

    # 添加均值为0、标准差为0.01的噪声
    y += torch.normal(0, 0.01, y.shape)
    # 将标签reshape为列向量
    return X, y.reshape((-1, 1))

def get_data_loader(features, labels, batch_size=32, shuffle=True):
    """创建数据加载器"""
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )