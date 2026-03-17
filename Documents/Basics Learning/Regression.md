
> 参考：https://zh-v2.d2l.ai/

## 介绍

回归（regression）是能为一个或多个自变量与因变量之间关系建模的一类方法。
在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系。

机器学习中的大多数任务都与**预测**有关。
当我们想预测一个**数值**时，就会涉及回归问题。

例子：
- 预测价格（房屋、股票）
- 预测住院时间
- 预测零售销量

注意：**并非所有预测都是回归问题**。

---

## 线性回归的基本元素

线性回归（linear regression）最简单且最流行，基于两个假设：
1. 自变量 $\mathbf{x}$ 与因变量 $y$ 是**线性关系**
2. 噪声服从**正态分布**

例子：根据房屋面积、房龄 预测 房价。

机器学习术语：
- 数据集：训练集（training set）
- 每行数据：样本（sample）
- 预测目标：标签（label）/目标（target）
- 预测依据：特征（feature）
- 样本数：$n$
- 第 $i$ 个样本：$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，标签 $y^{(i)}$

![](../Attachments/Basics/回归.png)

---

## 线性模型

目标（房价）可以表示为特征的加权和：
$$
\text{price} = w_{\text{area}} \cdot \text{area} + w_{\text{age}} \cdot \text{age} + b.
$$

- $w$：权重（weight），决定每个特征的影响
- $b$：偏置（bias/偏移量/截距）

即使特征不会为 0，**仍需要偏置项**，否则模型表达能力受限。

### 高维线性回归

$d$ 个特征时：
$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b.
$$

向量化表示：
$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b.
$$

批量形式（$n$ 个样本）：
$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b.
$$

训练目标：找到 $\mathbf{w}, b$，让预测尽可能接近真实标签。

---

## 损失函数

损失函数（loss function）量化**真实值与预测值的差距**。

回归最常用：**平方误差损失**
$$
l^{(i)}(\mathbf{w}, b) = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2.
$$

全局平均损失：
$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b)
$$

训练目标：
$$
(\mathbf{w}, b) = \arg\min_{\mathbf{w}, b}\ L(\mathbf{w}, b).
$$

---

## 解析解

线性回归是少数有**解析解（公式解）**的模型。

合并偏置后，最小化 $\|\mathbf{y}-\mathbf{X}\mathbf{w}\|^2$，解得：
$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}.
$$

解析解数学优美，但**限制极强**，无法用于深度学习。

---

## 随机梯度下降（SGD）

梯度下降（gradient descent）**几乎能优化所有深度学习模型**，沿损失递减方向更新参数。

实际使用 **小批量随机梯度下降（minibatch SGD）**：每次随机抽一小批数据计算梯度，更快更稳定。

---

### 更新公式

$$
\begin{aligned}
\mathbf{w} &\leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i\in\mathcal{B}} \mathbf{x}^{(i)}\left(\mathbf{w}^\top\mathbf{x}^{(i)}+b-y^{(i)}\right),\\
b &\leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i\in\mathcal{B}}\left(\mathbf{w}^\top\mathbf{x}^{(i)}+b-y^{(i)}\right).
\end{aligned}
$$

符号说明：
- $\mathcal{B}$：小批量样本集合
- $|\mathcal{B}|$：批量大小（batch size）
- $\eta$：学习率（learning rate）
- 批量大小、学习率：**超参数（hyperparameter）**

训练流程：
1. 随机初始化参数
2. 迭代抽取小批量，沿负梯度更新
3. 满足停止条件时停止

---

**标签**
#ML #BASICS #回归 #线性回归
