import sympy as sp

# 1. 定义符号
t = sp.symbols('t')
m, l, g = sp.symbols('m l g', real=True, positive=True)
theta = sp.Function('theta')(t)  # 关节角度随时间变化

# 定义速度 (theta_dot) 和 加速度 (theta_ddot)
theta_dot = theta.diff(t)
theta_ddot = theta_dot.diff(t)

# 2. 计算能量
# 动能 K = 1/2 * m * v^2。对于单摆，v = l * angular_velocity
v = l * theta_dot
K = sp.Rational(1, 2) * m * v**2

# 势能 P = m * g * h。假设最低点势能为0，则 h = l * (1 - cos(theta))
# 或者简单定义坐标原点在转轴，y轴向上，则 h = -l * cos(theta)
h = -l * sp.cos(theta)
P = m * g * h

# 3. 拉格朗日函数 L
L = K - P

# 4. 欧拉-拉格朗日方程
# d/dt (dL/d_theta_dot)
dL_dthetadot = sp.diff(L, theta_dot)
term1 = sp.diff(dL_dthetadot, t)

# dL/d_theta
term2 = sp.diff(L, theta)

# 力矩 tau = term1 - term2
tau = term1 - term2

# 5. 简化并打印结果
equation = sp.simplify(tau)
print("--- 自动推导的动力学方程 (Tau = ...) ---")
sp.pprint(equation)

print("\n--- 提取各项 ---")
# 对于单摆: M*q_ddot + G = 0 (无阻尼情况)
# 我们期望看到 M = m*l^2, G = m*g*l*sin(theta)
print(f"惯性项 (对应 theta_ddot): {equation.coeff(theta_ddot)}")
# 注意：sympy提取非线性项比较复杂，这里人工观察打印结果即可验证
# 理论结果应为: m * l**2 * theta_ddot + m * g * l * sin(theta)
