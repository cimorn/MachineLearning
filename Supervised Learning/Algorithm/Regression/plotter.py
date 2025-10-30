import matplotlib.pyplot as plt
import os

# 确保results文件夹存在，不存在则创建
os.makedirs('results', exist_ok=True)

def plot_scatter(features, labels, save_path=None):
    """
    绘制特征与标签的散点图，并可选择保存图片
    
    参数:
        features: 特征矩阵
        labels: 标签数据
        true_w: 真实权重
        true_b: 真实偏置
        save_path: 图片保存路径（如'./scatter_plot.png'），为None时不保存
    """
    plt.figure(figsize=(12, 5))
    
    # 第一个特征与标签
    plt.subplot(121)
    plt.scatter(features[:, 0].numpy(), labels.numpy(), s=8, alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Label')
    
    # 第二个特征与标签
    plt.subplot(122)
    plt.scatter(features[:, 1].numpy(), labels.numpy(), s=8, alpha=0.6)
    plt.xlabel('Feature 2')
    plt.ylabel('Label')
    
    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi控制清晰度
        print(f"散点图已保存至: {save_path}")


def plot_loss(losses, save_path=None):
    """绘制损失随训练轮次变化的折线图"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, len(losses))  # x轴范围从1开始
    plt.ylim(0, max(losses) * 1.1)  # y轴留一点余量

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi控制清晰度
        print(f"loss已保存至: {save_path}")


