# Changelog

## [1.0.0] - 2026-03-17

### ✨ 项目初始化
- 项目正式重构完成，包含基础机器学习、深度学习、强化学习三大模块
- 所有历史提交记录已清理，从当前版本开始全新记录

### 📁 项目结构
#### Basic Learning 基础机器学习
- **Classification**: 基于FashionMNIST数据集的图像分类算法实现
- **Regression**: 线性回归与深度回归模型实现

#### Reinforcement Learning 强化学习
- **DQN**: 经典深度Q网络算法实现，支持CartPole-v1、MountainCar-v0环境
- **DDQN**: 双重DQN算法实现，解决DQN过估计问题

### 📚 文档配置
- 统一文档目录`Documents/`，按算法分类存放所有Markdown格式文档
- 包含算法原理说明、实验结果分析和深度学习基础文档
- 提供`index.md`作为所有文档的统一入口索引
- 配置Git提交钩子，自动格式化提交信息为`YYMMDDHHMM: 标题`格式

### 🔧 开发规范
- 每个算法目录独立存放，含专属`requirements.txt`依赖清单
- 数据集`data/`和实验结果`results/`目录已加入`.gitignore`，避免大文件提交
- 代码与文档完全分离，代码目录仅保留运行相关文件
- 所有文档统一放在`Documents/`目录下，便于查找和维护
