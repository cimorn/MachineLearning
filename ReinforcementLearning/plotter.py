import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def smooth_curve(points, factor=0.9):
    """使用指数移动平均平滑曲线"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_learning_curve(returns, env_name, algo_name, save_path=None):
    """绘制单条训练回报曲线"""
    plt.figure(figsize=(10, 5))
    plt.title(f"{algo_name} Training on {env_name}")
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    
    # 绘制原始数据
    plt.plot(returns, alpha=0.3, color='blue', label='Raw')
    
    # 绘制平滑数据
    if len(returns) > 10:
        smoothed = smooth_curve(returns)
        plt.plot(smoothed, color='red', label='Smoothed')
        
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_return_comparison(return_dict, env_name, save_path=None):
    """
    绘制多种算法的训练曲线对比图
    :param return_dict: 字典 {'DQN': [r1, r2...], 'DDQN': [r1, r2...]}
    """
    plt.figure(figsize=(10, 5))
    plt.title(f"Training Comparison on {env_name}")
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Returns')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 蓝, 橙, 绿, 红
    
    for i, (algo_name, returns) in enumerate(return_dict.items()):
        color = colors[i % len(colors)]
        if len(returns) > 10:
            smoothed = smooth_curve(returns)
            plt.plot(smoothed, color=color, label=algo_name, linewidth=2)
        else:
            plt.plot(returns, color=color, label=algo_name, linewidth=2)
            
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"对比图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_eval_comparison(eval_dict, env_name, save_path=None):
    """
    绘制多种算法的评估结果对比柱状图 (带标准差误差棒)
    :param eval_dict: 字典 {'DQN': [score1, score2...], 'DDQN': [...]}
    """
    plt.figure(figsize=(8, 6))
    plt.title(f"Evaluation Comparison on {env_name}")
    plt.ylabel('Average Return')
    
    algos = list(eval_dict.keys())
    means = [np.mean(eval_dict[a]) for a in algos]
    stds = [np.std(eval_dict[a]) for a in algos]
    
    x_pos = np.arange(len(algos))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 绘制柱状图
    bars = plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, capsize=10, color=colors[:len(algos)])
    plt.xticks(x_pos, algos)
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱子上标记数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}', ha='center', va='bottom')

    if save_path:
        plt.savefig(save_path)
        print(f"评估对比图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()

def save_animation(frames, path, fps=30):
    """保存GIF动画"""
    if not frames:
        print("没有帧数据，无法保存动画。")
        return
        
    height, width, _ = frames[0].shape
    dpi = 72
    
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    img = ax.imshow(frames[0])
    
    def update(frame):
        img.set_data(frame)
        return [img]

    print(f"正在生成动画: {path} (共 {len(frames)} 帧)...")
    # 减少 interval 可以让动画变快，这里保持 fps 一致
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps)
    
    try:
        anim.save(path, writer='pillow', fps=fps)
        print(f"动画保存成功！")
    except Exception as e:
        print(f"动画保存失败: {e}")
    finally:
        plt.close()