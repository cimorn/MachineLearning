# Machine Learning Algorithms Collection

机器学习算法实现集合，包含基础机器学习、深度学习和强化学习三大类算法的完整实现，配套详细文档和实验结果。

## 📚 项目结构

```
MachineLearning/
├── BasicLearning/              # 基础机器学习算法
│   ├── Classification/          # 分类任务
│   └── Regression/              # 回归任务
├── ReinforcementLearning/       # 强化学习算法
│   ├── DQN/                     # DQN算法实现
│   ├── DDQN/                    # DDQN算法实现
│   ├── OPE/                     # 离线策略评估算法实现
│   └── results/                 # 公共实验结果目录
├── Documents/                   # 所有文档资源（Markdown格式）
├── .gitignore                   # Git忽略配置
├── CHANGELOG.md                 # 版本变更记录
└── README.md                    # 项目说明文档
```

## ✨ 已实现算法

### Basic Learning 基础机器学习
| 算法 | 任务 | 数据集 | 说明 |
|------|------|--------|------|
| CNN分类器 | 图像分类 | FashionMNIST | 基于卷积神经网络的服装图像分类 |
| 线性回归/深度回归 | 回归预测 | 合成数据集 | 对比线性模型和深度神经网络的回归效果 |

### Reinforcement Learning 强化学习
| 算法 | 支持环境 | 说明 |
|------|----------|------|
| DQN | CartPole-v1, MountainCar-v0 | 经典深度Q网络实现 |
| DDQN | CartPole-v1, MountainCar-v0 | 双重DQN算法，解决DQN过估计问题 |
| OPE | Pendulum-v1, Swimmer-v5 | 离线策略评估算法，包含倒立摆和Swimmer机器人两个完整实验环境和数据集，无需交互即可评估策略性能 |


## 🛠️ 环境安装

```bash
git clone https://github.com/cimorn/MachineLearning.git
cd MachineLearning
```

进入对应算法目录安装依赖：
```bash
cd <算法目录路径>
pip install -r requirements.txt
```

## 🚀 使用方法

进入对应算法目录直接运行主程序：
```bash
cd <算法目录路径>
python main.py
```

## 📝 目录规范

- 每个算法独立目录存放，包含专属`requirements.txt`依赖清单
- `data/`: 数据集目录（已加入.gitignore，无需提交）
- `results/`: 实验输出结果目录（已加入.gitignore，无需提交）
- 所有算法文档统一存放在根目录`Documents/`下（Markdown格式）
- 多实验算法（如OPE）按不同环境分子目录独立存放
