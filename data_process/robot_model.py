"""
机器人参数存储模块
"""
import numpy as np
from main_cali.DHRobotWithBeta_torch import DHRobotWithBeta, RevoluteDHWithBeta


def UR20_dh_params():
    """获取UR20机器人DH参数"""
    dh = [
        {'alpha': np.pi/2,   'theta': 0.0, 'd': 236.3, 'a': 0.0,    'beta': 0.0},  # Joint 1
        {'alpha': 0.0,       'theta': 0.0, 'd': 0.0,   'a': -862.0, 'beta': 0.0},  # Joint 2
        {'alpha': 0.0,       'theta': 0.0, 'd': 0.0,   'a': -728.7, 'beta': 0.0},  # Joint 3
        {'alpha': np.pi/2,   'theta': 0.0, 'd': 201.0, 'a': 0.0,    'beta': 0.0},  # Joint 4
        {'alpha': -np.pi/2,  'theta': 0.0, 'd': 159.3, 'a': 0.0,    'beta': 0.0},  # Joint 5
        {'alpha': 0.0,       'theta': 0.0, 'd': 154.3, 'a': 0.0,    'beta': 0.0},  # Joint 6
    ]
    return dh


def FR16_dh_params():
    """获取FR16机器人DH参数"""
    dh = [
        {'alpha': np.radians(90.0), 'theta': np.radians(-0.0), 'd': 180.0, 'a': -0.0, 'beta': 0.0},  # Joint 1
        {'alpha': np.radians(0.0), 'theta': np.radians(180.0), 'd': 0.0, 'a': 520.0, 'beta': 0.0},  # Joint 2
        {'alpha': np.radians(0.0), 'theta': np.radians(0.0), 'd': 0.0, 'a': 400.0, 'beta': 0.0},  # Joint 3
        {'alpha': np.radians(-90.0), 'theta': np.radians(-0.0), 'd': 159.0, 'a': 0.0, 'beta': 0.0},  # Joint 4
        {'alpha': np.radians(90.0), 'theta': np.radians(0.0), 'd': 114.0, 'a': 0.0, 'beta': 0.0},  # Joint 5
        {'alpha': np.radians(-0.0), 'theta': np.radians(180.0), 'd': 106.0, 'a': 0.0, 'beta': 0.0},  # Joint 6
    ]
    return dh


def create_robot(robot_name="FR16"):
    """
    根据机器人名称创建对应的机器人模型
    
    :param robot_name: 机器人名称 ("UR20" 或 "FR16")
    :return: 机器人模型
    """
    # 根据机器人名称获取对应的DH参数
    if robot_name.upper() == "UR20":
        dh_params = UR20_dh_params()
        robot_full_name = "UR20_Calibration_Robot"
    elif robot_name.upper() == "FR16":
        dh_params = FR16_dh_params()
        robot_full_name = "FR16_Calibration_Robot"
    else:
        raise ValueError(f"不支持的机器人类型: {robot_name}。支持的类型: UR20, FR16")
    
    # 创建连杆列表
    links = []
    for i, params in enumerate(dh_params):
        link = RevoluteDHWithBeta(
            a=params['a'],           # 连杆长度
            d=params['d'],           # 连杆偏移
            alpha=params['alpha'],   # 连杆扭角
            beta=params['beta'],     # 连杆扭角修正
            offset=params['theta'],  # 关节偏移角度(theta)
            name=f"link{i+1}"
        )
        links.append(link)
    
    # 创建机器人模型
    robot = DHRobotWithBeta(links, name=robot_full_name)
    
    return robot


