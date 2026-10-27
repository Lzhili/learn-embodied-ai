class PID:
    # 初始化：设置三个系数、目标值、采样周期
    def __init__(self, Kp, Ki, Kd, setpoint=0, sample_time=0.01):
        self.Kp = Kp                    # 比例系数 P（管快）
        self.Ki = Ki                    # 积分系数 I（管准）
        self.Kd = Kd                    # 微分系数 D（管稳）
        self.setpoint = setpoint        # 目标值（期望输出）
        self.sample_time = sample_time  # 采样周期（两次计算之间的间隔）

        self.prev_error = 0   # 上一次的误差，用于计算微分
        self.integral = 0     # 误差累计（积分项）

    # 每来一次测量值，就计算一次控制输出
    def update(self, measured_value):
        error = self.setpoint - measured_value  # 1. 计算误差 = 目标值 - 实际值
        self.integral += error * self.sample_time  # 2. 积分项：误差随时间累加
        derivative = (error - self.prev_error) / self.sample_time  # 3. 微分项：误差的变化速度

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative  # 4. 三项相加得到控制输出
        self.prev_error = error  # 保存本次误差，供下次计算微分用

        return output


pid = PID(Kp=1.0, Ki=0.1, Kd=0.01, setpoint=100)  # 目标值设为 100
measured_value = 90  # 假设的当前测量值
control_input = pid.update(measured_value)  # 计算一次控制量

print(f"Control Input: {control_input}")

