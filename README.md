# Machine Learning Algorithms Collection

机器学习算法实现集合，包含基础机器学习、深度学习和强化学习三大类算法的完整实现，配套详细文档和实验结果。

## 📚 项目结构

```
MachineLearning/
├── BasicLearning/              # 基础机器学习算法
│   ├── Classification/          # 分类任务
│   │   ├── data/                # 数据集（FashionMNIST）
│   │   └── *.py                 # 代码实现
│   └── Regression/              # 回归任务
│       └── *.py                 # 代码实现
├── ReinforcementLearning/       # 强化学习算法
│   ├── DQN/                     # DQN算法实现
│   │   └── *.py                 # 代码文件
│   ├── DDQN/                    # DDQN算法实现
│   │   └── *.py                 # 代码文件
│   └── results/                 # 公共实验结果目录
├── Documents/                        # 所有文档资源（Markdown格式）
│   ├── Basics Learning/
│   │   ├── Classification.md    # 分类算法文档
│   │   └── Regression.md        # 回归算法文档
│   ├── Deep Learning/
│   │   └── 神经网络.md          # 深度学习基础文档
│   ├── Reinforcement Learning/
│   │   ├── DQN.md               # DQN算法文档
│   │   └── DDQN.md              # DDQN算法文档
│   └── index.md                 # 文档索引
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

## 🛠️ 环境安装

```bash
# 克隆项目
git clone https://github.com/cimorn/MachineLearning.git
cd MachineLearning
```

进入对应算法目录安装所需依赖：

```bash
# 安装分类算法依赖
cd "Basic Learning/Classification"
pip install -r requirements.txt

# 安装回归算法依赖
cd "Basic Learning/Regression"
pip install -r requirements.txt

# 安装DQN算法依赖
cd ReinforcementLearning/DQN
pip install -r requirements.txt

# 安装DDQN算法依赖
cd ReinforcementLearning/DDQN
pip install -r requirements.txt
```

## 🚀 使用方法

进入对应算法目录运行主程序即可：

```bash
# 运行分类算法
cd "Basic Learning/Classification"
python main.py

# 运行回归算法
cd "Basic Learning/Regression"
python main.py

# 运行DQN算法
cd ReinforcementLearning/DQN
python main.py

# 运行DDQN算法
cd ReinforcementLearning/DDQN
python main.py
```

所有算法的详细原理说明和实验结果分析可查看`Docs/`目录下对应Markdown文档。

## 📝 目录规范

所有算法目录保持统一结构：
- 代码目录：仅存放算法实现源码
- `data/`: 数据集（已加入.gitignore，无需提交）
- `results/`: 实验输出结果（图片、日志等，已加入.gitignore）
- `Docs/`: 统一存放所有算法文档（Markdown格式）
