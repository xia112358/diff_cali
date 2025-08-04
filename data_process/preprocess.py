import os
import json
import numpy as np
import open3d as o3d
from typing import List, Optional, Tuple, Dict, Any


def read_ply_file(ply_file: str) -> Optional[np.ndarray]:
    """读取PLY点云文件
    
    Args:
        ply_file: PLY文件路径
        
    Returns:
        点云数据(N x 3的numpy数组)，如果读取失败返回None
    """
    try:
        # 使用Open3D读取PLY文件
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            print(f"❌ PLY文件 {ply_file} 为空或无法读取点云数据")
            return None
        
        # 过滤NaN和无穷值
        original_count = len(points)
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        filtered_count = len(points)
        if filtered_count < original_count:
            print(f"⚠️  过滤了 {original_count - filtered_count} 个包含NaN/Inf的点")
        
        if len(points) == 0:
            print(f"❌ PLY文件 {ply_file} 过滤后为空")
            return None
            
        print(f"✓ 成功读取PLY文件: {os.path.basename(ply_file)}, {len(points)} 个点")
        return points
        
    except Exception as e:
        print(f"❌ 读取PLY文件 {ply_file} 失败: {str(e)}")
        return None


def read_poses_json(json_file: str, convert_to_rad: bool = True) -> Optional[List[np.ndarray]]:
    """读取包含位姿关节角度的JSON文件
    
    Args:
        json_file: JSON文件路径
        convert_to_rad: 是否将角度制转换为弧度制，默认True
        
    Returns:
        关节角度列表，每个元素是6维的numpy数组
        
    Note:
        - 输入的关节角度默认为角度制
        - 如果convert_to_rad为True，函数会自动转换为弧度制
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            poses_data = json.load(f)
        
        joint_angles_list = []
        
        # 读取格式: {"poses": [{"joints": [q1,q2,q3,q4,q5,q6]}, ...]}
        if 'poses' in poses_data:
            for i, pose in enumerate(poses_data['poses']):
                if 'joints' in pose:
                    joint_angles = np.array(pose['joints'], dtype=np.float64)
                    
                    # 检查关节角度维度
                    if len(joint_angles) != 6:
                        print(f"❌ 位姿 {i} 的关节角度维度错误，应为6维，实际为{len(joint_angles)}维")
                        return None
                    
                    # 根据需要转换角度制到弧度制
                    if convert_to_rad:
                        joint_angles = np.deg2rad(joint_angles)
                    
                    joint_angles_list.append(joint_angles)
                else:
                    print(f"❌ JSON格式错误，位姿 {i} 应包含'joints'字段")
                    return None
        else:
            print(f"❌ JSON文件格式错误，应包含'poses'字段")
            return None
            
        print(f"✓ 成功读取位姿文件: {os.path.basename(json_file)}, {len(joint_angles_list)} 个位姿")
        return joint_angles_list
        
    except Exception as e:
        print(f"❌ 读取位姿文件 {json_file} 失败: {str(e)}")
        return None


def read_point_clouds(data_dir: str, num_clouds: int = 6, filename_pattern: str = "point_cloud_{:05d}.ply") -> List[Optional[np.ndarray]]:
    """批量读取点云文件
    
    Args:
        data_dir: 数据目录路径
        num_clouds: 要读取的点云文件数量
        filename_pattern: 文件名格式，默认为"point_cloud_{:05d}.ply"
        
    Returns:
        点云数据列表，每个元素是点云的numpy数组或None(如果读取失败)
    """
    point_clouds = []
    
    for i in range(1, num_clouds + 1):  # 修改起始索引为1
        ply_file = os.path.join(data_dir, filename_pattern.format(i))
        
        if not os.path.exists(ply_file):
            print(f"❌ 点云文件 {ply_file} 不存在")
            point_clouds.append(None)
            continue
            
        point_cloud = read_ply_file(ply_file)
        point_clouds.append(point_cloud)
    
    # 统计成功读取的点云数量
    successful_reads = sum(1 for pc in point_clouds if pc is not None)
    print(f"✓ 成功读取 {successful_reads}/{num_clouds} 个点云文件")
    
    return point_clouds


def load_dataset(data_dir: str, num_samples: int = 6, poses_file: str = "poses.json", 
                 clouds_subdir: str = "clouds", convert_to_rad: bool = True) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """加载完整数据集(位姿和点云)
    
    Args:
        data_dir: 数据目录路径
        num_samples: 要读取的样本数量
        poses_file: 位姿文件名
        clouds_subdir: 点云文件子目录名
        convert_to_rad: 是否将角度制转换为弧度制
        
    Returns:
        (poses, point_clouds): 位姿列表和点云列表的元组
        如果读取失败，对应部分返回None
    """
    print(f"从 {data_dir} 加载数据集...")
    
    # 读取位姿数据
    poses_json_file = os.path.join(data_dir, poses_file)
    poses = read_poses_json(poses_json_file, convert_to_rad=convert_to_rad)
    
    # 读取点云数据
    clouds_dir = os.path.join(data_dir, clouds_subdir)
    point_clouds = read_point_clouds(clouds_dir, num_samples)
    
    # 检查数据一致性
    if poses is not None and point_clouds is not None:
        poses_count = len(poses)
        clouds_count = sum(1 for pc in point_clouds if pc is not None)
        
        if poses_count != clouds_count:
            print(f"⚠️  位姿数量({poses_count})与点云数量({clouds_count})不匹配")
        else:
            print(f"✓ 数据集加载完成: {poses_count} 组位姿-点云对")
    
    return poses, point_clouds


def create_observations(poses: List[np.ndarray], point_clouds: List[np.ndarray]) -> List[Dict[str, Any]]:
    """创建观测数据列表，用于标定算法
    
    Args:
        poses: 位姿列表
        point_clouds: 点云列表
        
    Returns:
        观测数据列表，每个元素包含'joint_state'和'point_cloud'字段
    """
    observations = []
    
    for i, (pose, point_cloud) in enumerate(zip(poses, point_clouds)):
        if pose is not None and point_cloud is not None:
            observation = {
                'joint_state': pose,
                'point_cloud': point_cloud
            }
            observations.append(observation)
        else:
            print(f"⚠️  跳过第 {i} 组数据: 位姿或点云数据无效")
    
    print(f"✓ 创建了 {len(observations)} 个有效观测")
    return observations


# 示例用法
if __name__ == "__main__":
    # 设置数据路径
    data_dir = r"D:\桌面\实习\diff_cali\data"
    
    # 示例1: 读取单个文件
    print("=== 示例1: 读取单个文件 ===")
    poses = read_poses_json(os.path.join(data_dir, "poses.json"))
    point_cloud = read_ply_file(os.path.join(data_dir, "clouds", "point_cloud_00000.ply"))
    
    # 示例2: 批量读取指定数量的文件
    print("\n=== 示例2: 批量读取文件 ===")
    num_samples = 6  # 读取前3个文件
    poses, point_clouds = load_dataset(data_dir, num_samples=num_samples)
    
    # 示例3: 创建观测数据
    if poses is not None and point_clouds is not None:
        print("\n=== 示例3: 创建观测数据 ===")
        # 过滤掉None值
        valid_poses = [p for p in poses if p is not None]
        valid_clouds = [pc for pc in point_clouds if pc is not None]
        observations = create_observations(valid_poses, valid_clouds)
        
        # 显示数据信息
        if observations:
            print(f"第一个观测数据:")
            print(f"  关节角度: {observations[0]['joint_state']}")
            print(f"  点云形状: {observations[0]['point_cloud'].shape}")