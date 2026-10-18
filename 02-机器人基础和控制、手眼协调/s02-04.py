import sympy as sp
import numpy as np

def calculate_jacobian_2link():
    # 1. 定义符号
    theta1, theta2 = sp.symbols('theta1 theta2')
    l1, l2 = sp.symbols('l1 l2')

    # 2. 正运动学方程 (末端 x, y 与姿态角 phi)
    # x = l1*c1 + l2*c12
    # y = l1*s1 + l2*s12
    x = l1 * sp.cos(theta1) + l2 * sp.cos(theta1 + theta2)
    y = l1 * sp.sin(theta1) + l2 * sp.sin(theta1 + theta2)
    phi = theta1 + theta2

    # 位姿向量: [x, y, phi]^T
    pose = sp.Matrix([x, y, phi])
    
    # 关节变量向量
    q = sp.Matrix([theta1, theta2])

    # 3. 计算雅可比矩阵 J = d(pose)/d(q), 维度为 3x2
    J = pose.jacobian(q)

    print("--- 完整符号形式的雅可比矩阵 J (3x2) ---")
    sp.pprint(J)

    # 4. 数值验证 (奇异点测试)
    # 完整 J 为 3x2 非方阵，不能直接取 det(J)。
    # 对平面 2R 位置任务，常用其线速度子矩阵 Jv(前两行)判断奇异性。
    # 当 theta2 = 0 时 (手臂完全伸直)，Jv 行列式应为 0。
    J_func = sp.lambdify((theta1, theta2, l1, l2), J, 'numpy')
    
    # 测试: l1=1, l2=1, theta1=0, theta2=0 (伸直)
    J_val = J_func(0, 0, 1, 1)
    Jv_val = J_val[:2, :]
    det_Jv = np.linalg.det(Jv_val)
    rank_J = np.linalg.matrix_rank(J_val)
    
    print("\n--- 数值验证 (手臂伸直 theta2=0) ---")
    print(f"完整雅可比矩阵 J:\n{J_val}")
    print(f"线速度子矩阵 Jv:\n{Jv_val}")
    print(f"det(Jv) (接近0表示位置奇异): {det_Jv}")
    print(f"rank(J) = {rank_J} (用于判断完整雅可比列秩是否下降；小于2才表示列秩退化)")
    print("注: 位置任务的奇异性以 Jv 为准（例如 det(Jv) 是否接近 0）。")

    # 5. 力映射示例: tau = J^T * F
    # F = [fx, fy, mz]^T, 对应平面力与末端力矩
    F = np.array([10.0, 5.0, 1.0])
    tau = J_val.T @ F
    print("\n--- 力/力矩映射示例 ---")
    print(f"末端广义力 F = {F}")
    print(f"关节力矩 tau = J^T F = {tau}")

calculate_jacobian_2link()
