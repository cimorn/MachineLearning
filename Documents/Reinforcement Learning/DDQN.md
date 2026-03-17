# Double DQN

DQN 通过经验回放与目标网络缓解了训练不稳定问题，但仍存在**Q 值过估计（Overestimation Bias）**缺陷。

DQN 使用**同一目标网络**同时完成**动作选择**与**价值评估**，神经网络的逼近误差会被 $\max$ 操作放大，导致 Q 值系统性偏高，进而造成策略次优、收敛变慢。

Double DQN（DDQN）的核心思想源于 Double Q-Learning：**将动作选择与价值评估解耦，由两个独立网络分别承担，抑制误差累积。**



## DQN 与 DDQN 对比

| 对比维度 | DQN | DDQN |
|:-------:|:---:|:----:|
| 目标值计算 | 目标网络同时负责选动作与评估 | 在线网络选动作，目标网络评估 |
| 目标公式 | $y_t^\text{DQN} = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1},a')$ | $y_t^\text{DDQN} = r_t + \gamma Q_{\theta^-}\big(s_{t+1},\arg\max_{a'} Q_\theta(s_{t+1},a')\big)$ |
| 过估计风险 | 高 | 低 |
| 网络结构 | 训练网络 + 目标网络 | 在线网络 + 目标网络（结构相同） |
| 计算开销 | 基础 | 几乎无额外开销 |



## 核心机制：解耦

DDQN 仍采用双网络结构，但分工明确：
- **在线网络 $Q_\theta(s,a)$**：负责**动作选择**，即确定下一状态的最优动作 $a^* = \arg\max_{a'} Q_\theta(s_{t+1},a')$。
- **目标网络 $Q_{\theta^-}(s,a)$**：负责**价值评估**，对选定动作计算 Q 值，提供稳定目标。

由于两个网络独立优化，同时高估同一动作的概率极低，可有效抑制过估计。



## 目标值与损失函数

DDQN 损失函数形式与 DQN 一致，采用均方误差：
$$
L(\theta) = \frac{1}{N}\sum_{i=1}^N \big(Q_\theta(s_i,a_i) - y_i^\text{DDQN}\big)^2
$$

DDQN 目标值 $y_i^\text{DDQN}$：
$$
y_i^\text{DDQN} =
\begin{cases}
r_i & \text{若 } done_i = \text{True} \\
r_t + \gamma Q_{\theta^-}\big(s_{t+1},\arg\max_{a'} Q_\theta(s_{t+1},a')\big) & \text{否则}
\end{cases}
$$



## 算法流程

1. **初始化**
   - 随机初始化在线网络参数 $\theta$，复制得到目标网络 $\theta^- \leftarrow \theta$。
   - 初始化经验回放池 $R$、探索率 $\epsilon$。

2. **对每个回合 $e=1,\dots,E$**
   - 获取初始状态 $s_1$。
   - 对每一步 $t=1,\dots,T$：
     1. $\epsilon$-贪婪策略选择动作 $a_t$。
     2. 执行动作得到 $r_t, s_{t+1}, done$，存入经验回放池。
     3. 若回放池数据充足，随机采样批量数据：
        - 由在线网络选择最优动作 $a_i^*$。
        - 由目标网络计算 DDQN 目标 $y_i^\text{DDQN}$。
        - 计算 MSE 损失并更新在线网络 $\theta$。
     4. 每隔 $C$ 步更新目标网络：$\theta^- \leftarrow \theta$。
     5. 若终止则跳出当前回合，否则继续。
   - 衰减探索率 $\epsilon$。



## 优点与适用性

- 显著缓解 Q 值过估计，提升策略可靠性。
- 结构与 DQN 完全兼容，实现简单、无额外开销。
- 训练更稳定，收敛速度更快。
- 可与 PER、Dueling 等改进结合，是 Rainbow 等集成算法的基础组件。
- 适用于复杂动作空间、奖励稀疏环境，如游戏 AI、机器人控制等。

---

**标签**
#ML #RL #DL #DDQN #DQN