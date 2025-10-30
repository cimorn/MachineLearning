from data_loader import load_data
from model import FashionNN
from trainer import train_model, test_model, plot_losses, visualize_predictions

def main():
    # 1. 加载数据
    batch_size = 64
    train_loader, test_loader, label_names = load_data(batch_size)
    
    # 2. 初始化模型
    model = FashionNN()
    print("模型初始化完成")
    
    # 3. 训练模型
    epochs = 10
    print(f"开始训练，共{epochs}个epoch...")
    train_losses = train_model(model, train_loader, epochs=epochs)
    
    # 4. 测试模型
    print("开始测试...")
    test_accuracy = test_model(model, test_loader)
    
    # 5. 可视化结果
    plot_losses(train_losses)
    visualize_predictions(model, test_loader, label_names)

if __name__ == "__main__":
    main()