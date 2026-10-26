import numpy as np

def dh_matrix(theta, d, a, alpha):
    """
    根据标准DH参数计算变换矩阵 T_{i-1, i}
    单位: 角度使用弧度, 长度使用米
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def forward_kinematics_2link(joint_angles, link_lengths):
    """
    计算2连杆平面机械臂的末端位置
    """
    theta1, theta2 = joint_angles
    L1, L2 = link_lengths
    
    # 连杆 1: frame 0 -> frame 1
    # theta=q1, d=0, a=L1, alpha=0
    T_01 = dh_matrix(theta1, 0, L1, 0)
    
    # 连杆 2: frame 1 -> frame 2
    # theta=q2, d=0, a=L2, alpha=0
    T_12 = dh_matrix(theta2, 0, L2, 0)
    
    # 总变换矩阵
    T_02 = T_01 @ T_12
    
    return T_02

# --- 验证 ---
# 设定: L1=1m, L2=1m
# 情况 1: q1=0, q2=0 -> 手臂伸直横在X轴 -> 末端应为 (2, 0, 0)
# 情况 2: q1=90度, q2=0 -> 手臂垂直指天 -> 末端应为 (0, 2, 0)
# 情况 3: q1=0, q2=90度 -> 第一段横着，第二段向上 -> 末端应为 (1, 1, 0)

L_params = [1.0, 1.0]
q_case3 = [0, np.radians(90)] 

T_final = forward_kinematics_2link(q_case3, L_params)
pos_final = T_final[:3, 3] # 左上角是姿态（旋转矩阵），右上角 T[:3, 3] 是位置

print(f"关节角: {np.degrees(q_case3)}")
print(f"计算出的末端位置 (x, y, z): {np.round(pos_final, 4)}")
# 预期输出: x=1.0, y=1.0, z=0.0
print("------------------------------------")


# --- 05综合实战与仿真---UR5 六自由度机械臂正运动学 ---
def forward_kinematics(dh_params, joint_angles):
    """
    通用正运动学: 根据标准DH参数表计算末端执行器位姿 T_0n。
    dh_params: 列表, 每个元素为 (alpha, a, d)
    joint_angles: 各旋转关节的角度 theta (列表), 与 dh_params 一一对应
    """
    T = np.eye(4)
    for (alpha, a, d), theta in zip(dh_params, joint_angles):
        T = T @ dh_matrix(theta, d, a, alpha)
    return T

# --- UR5 标准DH参数 ---
# 每个元素顺序: (alpha, a, d)
ur5_dh = [
    ( np.pi/2,      0.0,  89.16),  # 关节1 (Base)
    ( 0.0,       -425.0,   0.0 ),  # 关节2 (Shoulder)
    ( 0.0,       -392.25,  0.0 ),  # 关节3 (Elbow)
    ( np.pi/2,      0.0, 109.15),  # 关节4 (Wrist 1)
    (-np.pi/2,      0.0,  94.65),  # 关节5 (Wrist 2)
    ( 0.0,          0.0,  82.30),  # 关节6 (Wrist 3)
]

# 零位: q = [0, 0, 0, 0, 0, 0]
q_zero = [0, 0, 0, 0, 0, 0]

T_06 = forward_kinematics(ur5_dh, q_zero)
pos_zero = T_06[:3, 3]

print(f"UR5 零位末端坐标 (x, y, z) mm: {np.round(pos_zero, 2)}")
# 预期输出: x=-817.25, y=-191.45, z=-5.49
