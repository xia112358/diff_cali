import torch
import os
from datetime import datetime
import json
from .DHRobotWithBeta_torch import DHRobotWithBeta

def generate_sphere_point_cloud_data(robot: DHRobotWithBeta, num_samples: int = 50, noise_std: float = 0.2, 
                                     save_data: bool = False, experiment_name: str = None):
    """
    生成基于球面点云的高精度校准数据

    :param robot: DHRobotWithBeta 实例
    :param num_samples: 样本数量 (增加到50个以上)
    :param noise_std: 噪声标准差 (0.2mm)
    :param save_data: 是否保存数据到文件
    :param experiment_name: 实验名称，用于文件命名
    :return: 生成的观测数据字典
    """
    print(f"Generating {num_samples} high-precision sphere-based point cloud samples...")

    # 设置球心在基坐标系中的固定位置和球半径
    sphere_center_base = torch.tensor([1500, 0, 500], dtype=torch.float64)  # 球心在基坐标系的位置 (mm)
    sphere_radius = 15  # 球的半径 (mm)

    # 生成观测数据
    obs_data = []

    for i in range(num_samples):
        # 生成随机关节角度
        q = []
        for j, link in enumerate(robot.links):
            if hasattr(link, 'qlim') and link.qlim is not None:
                q_val = torch.rand(1, dtype=torch.float64) * (link.qlim[1] - link.qlim[0]) + link.qlim[0]
                q_val = q_val.item()
            else:
                q_val = torch.rand(1, dtype=torch.float64) * (2 * torch.pi) - torch.pi  # 默认范围 [-π, π]
                q_val = q_val.item()
            q.append(q_val)

        # 计算前向运动学:传感器到基坐标系的变换
        T_sensor_to_base = robot.fkine(q)
        T_base_to_sensor = torch.inverse(T_sensor_to_base)

        # 生成球面点云
        point_cloud = []
        for _ in range(100):  # 每个位姿生成100个点
            theta = torch.rand(1, dtype=torch.float64) * (2 * torch.pi)  # 球面角度 theta [0, 2π]
            phi = torch.rand(1, dtype=torch.float64) * torch.pi         # 球面角度 phi [0, π]

            # 球面点的笛卡尔坐标
            x = sphere_radius * torch.sin(phi) * torch.cos(theta)
            y = sphere_radius * torch.sin(phi) * torch.sin(theta)
            z = sphere_radius * torch.cos(phi)

            # 球面点在基坐标系中的位置
            sphere_point_base = torch.stack([x.squeeze(), y.squeeze(), z.squeeze()]) + sphere_center_base

            # 将球面点从基坐标系转换到传感器坐标系
            sphere_point_base_homo = torch.cat([
                sphere_point_base,
                torch.tensor([1.0], dtype=torch.float64)  # 齐次坐标
            ])
            sphere_point_sensor = T_base_to_sensor @ sphere_point_base_homo
            sphere_point_sensor = sphere_point_sensor[:3]  # 取前3个元素

            # 添加噪声
            noise = torch.normal(mean=0.0, std=noise_std, size=(3,), dtype=torch.float64)
            sphere_point_sensor_noisy = sphere_point_sensor + noise

            point_cloud.append(sphere_point_sensor_noisy)  # 保持为 torch.Tensor 类型

        # 创建观测数据
        obs_data.append({
            "joint_state": torch.tensor(q, dtype=torch.float64),  # 转换为 torch.Tensor
            "point_cloud": torch.stack(point_cloud)  # 转换为 torch.Tensor
        })

    # 保存数据到文件
    if save_data:
        save_observations_to_file(obs_data, robot, num_samples, noise_std, experiment_name)

    return obs_data


def save_observations_to_file(obs_data, robot, num_samples, noise_std, experiment_name=None):
    """
    将观测数据和元数据保存到同一个文件
    
    :param obs_data: 观测数据列表
    :param robot: 机器人模型
    :param num_samples: 样本数量
    :param noise_std: 噪声标准差
    :param experiment_name: 实验名称
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建数据目录
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    robot_name = getattr(robot, 'name', 'robot')
    exp_name = experiment_name if experiment_name else "sphere_calibration"
    filename = f"{robot_name}_{exp_name}_{timestamp}.pt"
    
    # 创建包含数据和元数据的完整数据包
    data_package = {
        'observations': obs_data,  # 观测数据
        'metadata': {
            'experiment_info': {
                'experiment_name': exp_name,
                'robot_name': robot_name,
                'timestamp': timestamp,
                'description': 'Robot calibration sphere point cloud observations'
            },
            'data_info': {
                'num_observations': len(obs_data),
                'num_points_per_cloud': obs_data[0]['point_cloud'].shape[0] if obs_data else 0,
                'joint_dof': obs_data[0]['joint_state'].shape[0] if obs_data else 0,
                'noise_std': noise_std,
                'data_format': 'pytorch_tensor'
            },
            'sphere_params': {
                'center_base': [142, 20, 178],  # 球心在基坐标系的位置
                'radius': 15  # 球半径
            },
            'robot_info': {
                'num_joints': robot.n,
                'has_base_transform': hasattr(robot, 'base') and robot.base is not None,
                'has_tool_transform': hasattr(robot, 'tool') and robot.tool is not None
            }
        }
    }
    
    # 保存到单个文件
    file_path = os.path.join(data_dir, filename)
    torch.save(data_package, file_path)
    
    print(f"✅ 数据已保存到单个文件:")
    print(f"  文件路径: {file_path}")
    print(f"  文件名: {filename}")
    print(f"  样本数量: {len(obs_data)}")
    print(f"  每个点云包含: {obs_data[0]['point_cloud'].shape[0]} 个点")
    print(f"  文件大小: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    return file_path


def load_observations_from_file(data_file):
    """
    从单个文件加载观测数据和元数据
    
    :param data_file: 数据文件路径
    :return: (观测数据列表, 元数据字典)
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    data_package = torch.load(data_file)
    
    # 兼容旧格式（只有观测数据）
    if isinstance(data_package, list):
        obs_data = data_package
        metadata = None
        print(f"✅ 从文件加载了 {len(obs_data)} 个观测数据: {data_file}")
        print("⚠️  这是旧格式文件，没有元数据")
        return obs_data, metadata
    
    # 新格式（包含观测数据和元数据）
    obs_data = data_package['observations']
    metadata = data_package['metadata']
    
    print(f"✅ 从文件加载了 {len(obs_data)} 个观测数据: {data_file}")
    if metadata:
        print(f"  实验名称: {metadata['experiment_info']['experiment_name']}")
        print(f"  机器人: {metadata['experiment_info']['robot_name']}")
        print(f"  时间戳: {metadata['experiment_info']['timestamp']}")
        print(f"  噪声标准差: {metadata['data_info']['noise_std']}")
    
    return obs_data, metadata


def list_observation_files(data_dir=None):
    """
    列出所有可用的观测数据文件
    
    :param data_dir: 数据目录路径，默认为当前目录下的data文件夹
    :return: 数据文件列表和对应的元数据
    """
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
    
    if not os.path.exists(data_dir):
        print("数据目录不存在")
        return []
    
    # 查找所有.pt文件
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    file_info = []
    for data_file in data_files:
        data_path = os.path.join(data_dir, data_file)
        
        info = {
            'data_file': data_path,
            'filename': data_file,
            'size_mb': os.path.getsize(data_path) / (1024 * 1024)
        }
        
        # 尝试读取元数据
        try:
            data_package = torch.load(data_path)
            if isinstance(data_package, dict) and 'metadata' in data_package:
                info['metadata'] = data_package['metadata']
                info['has_metadata'] = True
            else:
                info['metadata'] = None
                info['has_metadata'] = False
        except:
            info['metadata'] = None
            info['has_metadata'] = False
        
        file_info.append(info)
    
    # 按文件名排序（最新的在前）
    file_info.sort(key=lambda x: x['filename'], reverse=True)
    
    return file_info