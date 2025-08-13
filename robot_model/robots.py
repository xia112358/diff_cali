"""
机器人参数存储模块 - 适配新的统一接口
"""
import numpy as np
from .mdh_robot import DHRobotWithBeta, RevoluteDHWithBeta



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
        {'alpha': np.radians(0.0), 'theta': np.radians(0.0), 'd': 0.0, 'a': -520.0, 'beta': 0.0},  # Joint 2
        {'alpha': np.radians(0.0), 'theta': np.radians(0.0), 'd': 0.0, 'a': -400.0, 'beta': 0.0},  # Joint 3
        {'alpha': np.radians(90.0), 'theta': np.radians(-0.0), 'd': 159.0, 'a': 0.0, 'beta': 0.0},  # Joint 4
        {'alpha': np.radians(-90.0), 'theta': np.radians(0.0), 'd': 114.0, 'a': 0.0, 'beta': 0.0},  # Joint 5
        {'alpha': np.radians(-0.0), 'theta': np.radians(0.0), 'd': 106.0, 'a': 0.0, 'beta': 0.0},  # Joint 6
    ]
    return dh


def ABB6700_dh_params():
    """获取ABB6700机器人DH参数"""
    dh = [
        {'alpha': np.deg2rad(-90.0), 'theta': np.deg2rad(0.0),   'd': 780.000,  'a': 320.000,   'beta': 0.0},  # Joint 1
        {'alpha': np.deg2rad(0.0),   'theta': np.deg2rad(-90.0), 'd': 0.000,    'a': 1125.000,  'beta': 0.0},  # Joint 2
        {'alpha': np.deg2rad(-90.0), 'theta': np.deg2rad(0.0),   'd': -0.000,   'a': 200.000,   'beta': 0.0},  # Joint 3
        {'alpha': np.deg2rad(-90.0), 'theta': np.deg2rad(-180.0),'d': 1392.500, 'a': 0.000,     'beta': 0.0},  # Joint 4
        {'alpha': np.deg2rad(-90.0), 'theta': np.deg2rad(-180.0),'d': 0.000,    'a': -0.000,    'beta': 0.0},  # Joint 5
        {'alpha': np.deg2rad(-0.000),'theta': np.deg2rad(180.0), 'd': 200.000,  'a': -0.000,    'beta': 0.0},  # Joint 6
    ]
    return dh


def FR16_calibrated_dh_params():
    """获取FR16机器人校准后的DH参数"""
    dh = [
        {'alpha': 1.570431, 'theta': 0.000000, 'd': 180.000, 'a': -0.949, 'beta': 0.000000},  # Joint 1
        {'alpha': -0.001079, 'theta': -0.002044, 'd': 0.000, 'a': -520.121, 'beta': -0.000801},  # Joint 2
        {'alpha': 0.000855, 'theta': -0.001400, 'd': 0.000, 'a': -399.701, 'beta': -0.000730},  # Joint 3
        {'alpha': 1.570837, 'theta': -0.001248, 'd': 159.000, 'a': 0.578, 'beta': 0.000000},  # Joint 4
        {'alpha': -1.571994, 'theta': 0.004534, 'd': 114.506, 'a': -0.125, 'beta': 0.000000},  # Joint 5
        {'alpha': -0.000000, 'theta': 0.000000, 'd': 106.000, 'a': 0.000, 'beta': 0.000000},  # Joint 6
    ]
    return dh


def create_robot_model(robot_name="FR16", model_type="mdh"):
    """
    根据机器人名称和模型类型创建对应的机器人模型
    
    :param robot_name: 机器人名称 ("UR20"、"FR16" 或 "ABB6700")
    :param model_type: 模型类型 ("mdh" 为修正DH参数模型)
    :return: 机器人模型
    """
    robot_name = robot_name.upper()
    
    if model_type.lower() == "mdh":
        # 修正DH参数模型
        if DHRobotWithBeta is None or RevoluteDHWithBeta is None:
            raise ImportError("Could not import required robot classes. Please ensure mdh_robot.py is available.")
        
        # 根据机器人名称获取对应的DH参数
        if robot_name == "UR20":
            dh_params = UR20_dh_params()
            robot_full_name = "UR20_Calibration_Robot"
        elif robot_name == "FR16":
            dh_params = FR16_dh_params()
            robot_full_name = "FR16_Calibration_Robot"
        elif robot_name == "FR16_CALIBRATED":
            dh_params = FR16_calibrated_dh_params()
            robot_full_name = "FR16_Calibrated_Robot"
        elif robot_name == "ABB6700":
            dh_params = ABB6700_dh_params()
            robot_full_name = "ABB6700_Calibration_Robot"
        else:
            raise ValueError(f"不支持的机器人类型: {robot_name}。支持的类型: UR20, FR16, FR16_CALIBRATED, ABB6700")
        
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
        # 设置机器人类型为 mdh，与统一接口保持一致
        if hasattr(robot, 'robot_type'):
            robot.robot_type = "mdh"
        return robot
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: mdh")


def create_robot(robot_name="FR16", model_type="mdh"):
    """
    根据机器人名称和模型类型创建对应的机器人模型 (向后兼容版本)
    
    :param robot_name: 机器人名称 ("UR20"、"FR16" 或 "ABB6700")
    :param model_type: 模型类型 ("mdh" 为修正DH参数模型)
    :return: 机器人模型
    """
    # 向后兼容：使用新的实现
    return create_robot_model(robot_name, model_type)




