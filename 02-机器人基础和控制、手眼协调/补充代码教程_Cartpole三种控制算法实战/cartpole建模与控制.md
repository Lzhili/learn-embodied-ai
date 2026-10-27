# Cartpole建模与PID、LQR、MPC控制算法的Python实现

### CartPole问题的目标是保持杆子竖直（角度为0）和小车在轨道中心（位置为0）。

教程中不涉及公式推导，如需了解PID\LQR\MPC的底层原理，可自行学习。

## 一、所用环境和基础
1. Windows
2. 矩阵论与控制理论
3. 环境安装见requirments.txt

## 二、Cartpole建模

Cartploe官方的动力学模型定义部分代码如下所示，参考http://incompleteideas.net/sutton/book/code/pole.c

详细可参考code中的cartpole_env.py文件，在实践过程中，为了适应线性控制器，我们对动力学模型进行了近似处理，将于后面进行介绍。

```python
# temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
# thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
# xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
```

转换成公式可以表示为：

$$
\begin{cases}
\ddot{x}=\frac{F+m_{p} l \dot{\theta}^{2} \sin \theta}{m_{p}+m_{c}}-\frac{m_{p} l \dot{\theta}^{2} \sin \theta}{m_{p}+m_{c}} \\
\ddot{\theta}=\frac{g \sin \theta-\cos \theta \cdot \frac{F+m_{p} l \dot{\theta}^{2} \sin \theta}{m_{p}+m_{c}}}{l\left(\frac{4}{3}-\frac{m_{p} \cos \theta^{2}}{m_{p}+m_{c}}\right)}
\end{cases}
$$



针对线性控制器，由于倒立摆的 $\theta$ 角很小，可以做近似处理，因此将 $\theta$ 趋近于0，同时 $\sin \theta$ 和 $\cos \theta$ 也做近似处理，可以得到， $\sin \theta \approx \theta$ ， $\cos \theta \approx 1$ ，经过近似处理后，公式可以表示为：

$$
\begin{cases}
\ddot{x}=\left(\frac{1}{m_{p}+m_{c}}+\frac{3 m_{p}}{m_{p}+4 m_{c}}\right) F-\frac{3 m_{p} g}{m_{p}+4 m_{c}} \theta \\
\ddot{\theta}=\frac{3 g\left(m_{p}+m_{c}\right) \theta}{l\left(m_{p}+4 m_{c}\right)}-\frac{3 F}{l\left(m_{p}+4 m_{c}\right)}
\end{cases}
$$


将上式整理为状态空间方程，可得：

$$
\left[\begin{array}{c}
\dot{x} \\
\ddot{x} \\
\dot{\theta} \\
\ddot{\theta}
\end{array}\right]=\left[\begin{array}{cccc}
0 & 1 & 0 & 0 \\
0 & 0 & -\frac{3 m_{p} g}{m_{p}+4 m_{c}} & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & \frac{3 g\left(m_{p}+m_{c}\right)}{l\left(m_{p}+4 m_{c}\right)} & 0
\end{array}\right]\left[\begin{array}{c}
x \\
\dot{x} \\
\theta \\
\dot{\theta}
\end{array}\right]+\left[\begin{array}{c}
0 \\
\frac{1}{m_{p}+m_{c}}+\frac{3 m_{p}}{m_{p}+4 m_{c}} \\
0 \\
-\frac{3}{l\left(m_{p}+4 m_{c}\right)}
\end{array}\right] F
$$

状态空间方程在LQR和MPC代码中使用，对应的python代码如下所示：
```python
# 离散化状态空间方程：x(k+1) = A·x(k) + B·u(k)
# 状态向量顺序为 [x, x_dot, theta, theta_dot]，u 是施加在小车上的力 F
# 符号对照：m=杆质量，M=车质量，L=杆长（对应公式里的 m_p、m_c、l）
# A = I + del_t*A_连续：用前向欧拉法把连续方程离散成小步长形式
A = np.eye(4) + del_t * np.array([[0, 1, 0, 0],                     # x_dot = x_dot（位置导数=速度）
                [0, 0, (-3*m*g)/(m+4*M), 0],                        # theta 对 x 加速度的影响
                [0, 0, 0, 1],                                       # theta_dot = theta_dot（角度导数=角速度）
                [0, 0, (3*(M+m)*g)/(L*(m+4*M)), 0]])                # theta 对角加速度的影响
# B 矩阵：输入力 F 进入各状态的方式
B = del_t * np.array([[0],                                          # 位置 x 不直接受 F 影响
                [(1/(M+m))+((3*m)/(m+4*M))],                        # F 对 x 加速度的贡献
                [0],                                                # 角度 theta 不直接受 F 影响
                [-(3)/(L*(m+4*M))]])                                # F 对角加速度的贡献（负号）
```

## 三、代码讲解
### 1. PID控制器

代码中仅采用PD控制，其中，分别对cart的位置和pole的角度计算偏差，并进行PD控制，PD控制的核心代码如下所示：

```python
def pid_cart(self, position):
    # 小车位置 PD 控制：目标是让小车回到轨道中心（position=0）
    bias = position                     # 位置偏差 = 当前位置 - 目标(0)
    d_bias = bias - self.bias_cart_1    # 偏差的变化量（近似微分）
    balance = self.kp_cart * bias + self.kd_cart * d_bias  # P项 + D项，得到该给车的力
    self.bias_cart_1 = bias             # 保存本次偏差，供下次算微分
    return balance

def pid_pole(self, angle):
    # 杆子角度 PD 控制：目标是让杆子竖直（angle=0）
    bias = angle                        # 角度偏差
    d_bias = bias - self.bias_pole_1    # 偏差的变化量（近似微分）
    # 注意负号：杆子往右倒（angle>0）要往左推（力为负），方向相反
    balance = -self.kp_pole * bias - self.kd_pole * d_bias
    self.bias_pole_1 = bias             # 保存本次偏差
    return balance
```

### 2. LQR控制器

离散LQR控制器问题描述：

$$
\begin{array}{l}
\text { S.T. } \quad x(k+1)=A x(k)+B u(k),\\
J=\frac{1}{2} \sum_{k=k_{0}}^{\infty}\left[x^{T}(k) Q x(k)+u^{T}(k) R u(k) + 2x^{T}(k)Nu_{k}\right]
\end{array}
$$

离散LQR控制器的计算步骤为：

1） 求解S矩阵：

$$
A^{T} S A-S-\left(A^{T} S B+N\right)\left(B^{T} S B+R\right)^{-1}\left(B^{T} S A+N^{T}\right)+Q=0
$$

2） 计算反馈矩阵K：

$$
K=R^{-1}B^{T}P
$$

3） 计算控制输入u：

$$
u=-K(x_{now}-x_{des})
$$

在LQR控制器实际的Cartpole使用过程中， $N$ 是状态与控制输入的交叉权重矩阵，Cartpole问题中没有需要特别惩罚这种交叉项，因此将其设为零矩阵。

LQR控制器的核心代码如下所示：

```python
def control_output(self, state_des, state_now):
    # 1. 解离散黎卡提方程，得到代价矩阵 S（LQR 的核心）
    S = np.matrix(linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
    # 2. 由 S 计算最优反馈增益矩阵 K
    K = np.matrix(linalg.inv(self.B.T * S * self.B + self.R) * (self.B.T * S * self.A))
    # print("state_now", state_now)
    # print("state_des", state_des)
    # 3. 反馈控制律 u = -K*(x - x_des)，-10 是动作缩放系数
    action_list  = -10*np.dot(K, state_now - state_des)
    action = action_list[0,0]  # 取出标量动作
    return action
```

其中计算黎卡提方程采用的是linalg库中的solve_discrete_are函数。

### 3. MPC控制器

MPC的核心组成部分：

1） 预测模型：用于根据当前状态和未来的控制输入，预测系统未来的状态轨迹。

2） 滚动优化：寻找一系列控制输入，使得定义的性能指标（代价函数）在预测时域内最优。

3） 反馈校正：基于测量对模型预测进行修正，并在下一个时间点根据新的状态重新进行预测和优化。

线性MPC的问题描述：

$$
\begin{array}{l}
\text { S.T. } \quad x(k+j+1 \mid k)=A x(k+j \mid k)+B u(k+j \mid k), \quad j=0,1, \ldots, N-1,\\
J=\sum_{j=0}^{N-1}\left[x^{T}(k+j \mid k) Q x(k+j \mid k)+u^{T}(k+j \mid k) R u(k+j \mid k)\right]+x^{T}(k+N \mid k) F x(k+N \mid k)
\end{array}
$$

其中：

* $N$ 是预测步长（预测区间）。
* $x(k+j \mid k)$ 是在 $k$ 时刻对 $k+j$ 时刻状态的预测。
* $u(k+j \mid k)$ 是在 $k$ 时刻计划在 $k+j$ 时刻的控制输入。
* $Q$ 是状态权重矩阵
* $R$ 是控制输入权重矩阵
* $F$ 是终端状态权重矩阵

MPC控制器的核心代码如下所示：

1）构建QP问题的矩阵：目的就是计算QP公式中的矩阵H和E

```python
def get_QPMatrix(A, B, Q, R, F, N):
    # 把 MPC 的预测+优化整理成二次规划(QP)的标准目标函数：
    #   J = 0.5 * U^T H U + U^T E x_k  （常数项略去）
    # 返回 H（二次项系数）和 E（一次项系数）

    # M：自由响应矩阵，M 的第 i 块把当前状态 x_k 传到第 i 步（x(k+i) = A^i x_k）
    M = np.vstack([np.eye(n), np.zeros((N*n, n))])
    # C：控制矩阵（类 Toeplitz），把未来 N 步控制序列 U 映射到状态轨迹
    C = np.zeros(((N+1)*n, N*p))
    temp = np.eye(n)  # 累计 A^i
    for i in range(1,N+1):
        rows = i * n + np.arange(n)          # 第 i 步对应的行块
        C[rows,:] = np.hstack([temp @ B, C[rows-n, :-p]])  # 递推填充 C
        temp = A @ temp                      # temp = A^i
        M[rows,:] = temp                     # 填充 M 的第 i 块

    # 把单步权重 Q、R 沿预测时域 N 步复制成块对角矩阵
    Q_ = np.kron(np.eye(N), Q)
    rows_Q, cols_Q = Q_.shape
    rows_F, cols_F = F.shape
    # Q_bar：块对角 [Q, Q, ..., Q, F]，最后一块是终端权重 F
    Q_bar = np.zeros((rows_Q+rows_F, cols_Q+cols_F))
    Q_bar[:rows_Q, :cols_Q] = Q_
    Q_bar[rows_Q:, cols_Q:] = F
    R_bar = np.kron(np.eye(N), R)

    # G = M.T @ Q_bar @ M  # 只含 x_k 的常数项，对求最优无影响，略去
    E = C.T @ Q_bar @ M                # 一次项系数（x_k 与 U 的交叉项）
    H = C.T @ Q_bar @ C + R_bar        # 二次项系数（U 的二次项 + 控制量惩罚）
    return E, H
```

2）在线求解MPC

CasADi: 这是一个强大的符号计算和自动微分框架，常用于解决最优控制和非线性优化问题。

```python
def mpc_prediction(x_k, E, H, N, p):
    # 在线求解 MPC：给定当前状态 x_k 和 QP 矩阵 E、H，求最优控制序列
    # 定义优化变量：未来 N 步的控制序列 U（每步 p 维）
    U = ca.SX.sym('U', N * p)
    # 目标函数：0.5*U^T H U + U^T E x_k（二次规划，最小值即最优控制）
    objective = 0.5 * ca.mtimes([U.T, H, U]) + ca.mtimes([U.T, E, x_k])
    qp = {'x': U, 'f': objective}  # 待优化变量与目标函数
    opts = {'print_time': False, 'ipopt': {'print_level': 0}}  # 关闭求解器打印
    solver = ca.nlpsol('solver', 'ipopt', qp, opts)  # 用 IPOPT 求解 QP

    # 求解问题
    sol = solver()
    # 提取最优解（完整未来控制序列）
    U_k = sol['x'].full().flatten()
    u_k = 10*U_k[:p]  # 滚动优化：只取第一步作为当前动作，10 是缩放

    return u_k
```

## 四、代码结果

### 1. PID控制结果

<img src="code/PID_CartPole-v1.gif"/>

<img src="code/PID.png"/>

### 2. LQR控制结果

![](code/LQR_CartPole-v1.gif)

![](code/LQR.png)

### 3. MPC控制结果

<img src="code/MPC_CartPole-v1.gif"/>

<img src="code/MPC.png"/>



