from robot_model.robots import create_robot_model
import torch

robot= create_robot_model("FR16_CALIBRATED", "mdh")

q_test = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]  # 6个关节角度

T_cumulative = torch.eye(4, dtype=torch.float64)  # 累积变换矩阵，初始为单位矩阵

print("基座变换矩阵:")
print(robot.base)
T_cumulative = T_cumulative @ robot.base

for i in range(robot.n):
    # 计算第i个关节的变换矩阵
    q_tensor = torch.tensor(q_test[i], dtype=torch.float64)
    T_joint = robot.links[i].A(q_tensor)  # 注意这里是links[i]不是links[i+1]
    
    print(f"关节{i+1} (角度={q_test[i]:.3f}rad) 变换矩阵:")
    print(T_joint)
    print()
    
    # 累积变换
    T_cumulative = T_cumulative @ T_joint  # 使用@而不是*进行矩阵乘法
    
    print(f"从基座到关节{i+1}的累积变换矩阵:")
    print(T_cumulative)
    position = T_cumulative[:3, 3]  # 提取位置向量
    print(f"累积位置: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
    print("-" * 60)