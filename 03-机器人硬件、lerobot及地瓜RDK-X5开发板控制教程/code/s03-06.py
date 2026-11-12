import numpy as np
import matplotlib.pyplot as plt

def complementary_filter_demo():
    # ---------- 1. 生成“真实角度” ----------
    # 0~10 s 取 1000 个点 → 采样频率约 100 Hz
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]                      # 采样周期 Δt ≈ 0.01 s

    # 真实运动：0.5 Hz 正弦摆动，幅度 10°（图中红线）
    #   θ(t) = A·sin(ωt),  A = 10°,  ω = 2π×0.5 = π rad/s
    true_angle = 10 * np.sin(2 * np.pi * 0.5 * t)

    # ---------- 2. 模拟两个传感器 ----------
    # 陀螺仪测角速度 θ'(t) = A·ω·cos(ωt)
    # 其中 10 * 2π * 0.5 就是 A·ω = 10π ≈ 31.4 °/s，所以单位是 °/s
    true_gyro = 10 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)

    # 真实陀螺总有零偏；这里简化成恒定 +0.5 °/s
    # 注意没加白噪声，所以纯积分曲线会非常“光滑”，只剩线性漂移
    gyro_reading = true_gyro + 0.5

    # 加速度计静置时可由重力分量反算倾角
    # 这里简化为“真值 + 高斯噪声”(σ=2°)，并假设没有线加速度干扰
    np.random.seed(42)                    # 【新增】固定随机数，便于复现
    acc_reading = true_angle + np.random.normal(0, 2.0, len(t))

    # ---------- 3. 互补滤波 ----------
    # 递推式：θ̂_k = α·(θ̂_{k-1} + ω_k·Δt) + (1-α)·θ_acc,k
    #   ├─ 第一项：陀螺积分，负责“快”和“平滑”
    #   └─ 第二项：加速度计观测，负责“准”和“不漂”
    fused_angle = []
    angle_est = 0.0        # 初始估计；t=0 真值恰为 0，所以没有明显启动瞬态
    alpha = 0.98           # 陀螺权重，0.95~0.99 常用
                           # 等效时间常数 τ ≈ Δt/(1-α) ≈ 0.5 s

    for i in range(len(t)):
        # (a) 预测：用陀螺角速度积分一步（前向欧拉）
        gyro_rate = gyro_reading[i]               # °/s
        angle_pred = angle_est + gyro_rate * dt   # °

        # (b) 观测：加速度计给出的绝对倾角
        acc_obs = acc_reading[i]

        # (c) 融合：98% 信预测，2% 信观测
        #     → 观测不直接决定输出，只做缓慢修正
        angle_est = alpha * angle_pred + (1 - alpha) * acc_obs
        fused_angle.append(angle_est)

    # ---------- 4. 画图对比 ----------
    plt.figure(figsize=(12, 6))

    # 加速度计原始读数：均值对，但抖动大
    plt.plot(t, acc_reading, 'g-', alpha=0.3, label='Accelerometer (Noisy)')

    # 纯陀螺积分：np.cumsum 做矩形积分 θ ≈ Σω·Δt
    # 恒定零偏 0.5 °/s，10 s 后约漂 5°
    gyro_only = np.cumsum(gyro_reading) * dt
    plt.plot(t, gyro_only, 'k--', alpha=0.5, label='Gyro Only (Drift)')

    # 互补滤波结果：目标是“既平滑又不漂”
    plt.plot(t, fused_angle, 'b-', linewidth=2, label='Fused Result (Complementary)')

    # 真值
    plt.plot(t, true_angle, 'r--', linewidth=1, label='True Angle')

    plt.title('Sensor Fusion: Complementary Filter Demo')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("滤波完成：请观察蓝色曲线如何兼顾了绿色的'绝对位置'和陀螺仪的'平滑性'")

if __name__ == "__main__":
    complementary_filter_demo()
