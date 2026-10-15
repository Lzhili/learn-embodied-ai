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
pos_final = T_final[:3, 3]

print(f"关节角: {np.degrees(q_case3)}")
print(f"计算出的末端位置 (x, y, z): {np.round(pos_final, 4)}")
# 预期输出: x=1.0, y=1.0, z=0.0
