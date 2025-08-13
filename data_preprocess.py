import os
import json
import torch
import numpy as np
import re
from typing import Dict, List, Any, Tuple
from data_process.preprocess import load_dataset, create_observations
from data_process.cloud import fit_sphere_center_TLS_A


def detect_poses_file_format(data_dir: str) -> Tuple[str, str]:
    """
    自动检测位姿文件格式
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        (file_format, file_path) 元组
        file_format: "txt", "json", 或 "none"
        file_path: 检测到的文件路径
    """
    # 检查txt格式文件
    poses_txt_path = os.path.join(data_dir, "poses.txt")
    if os.path.exists(poses_txt_path):
        return "txt", poses_txt_path
    
    # 检查json格式文件
    poses_json_path = os.path.join(data_dir, "poses.json")
    if os.path.exists(poses_json_path):
        return "json", poses_json_path
    
    return "none", ""


def load_dataset_auto_format(data_dir: str, num_samples: int = None, 
                           clouds_subdir: str = "clouds") -> Tuple[torch.Tensor, List]:
    """
    自动检测文件格式并加载数据集
    
    Args:
        data_dir: 数据目录路径
        num_samples: 要加载的样本数量，None表示加载全部
        clouds_subdir: 点云文件子目录名
        
    Returns:
        (poses, point_clouds) 元组
        poses: torch.Tensor，形状为 (N, 6)
        point_clouds: 点云数据列表
    """
    try:
        # 自动检测文件格式
        file_format, file_path = detect_poses_file_format(data_dir)
        
        if file_format == "none":
            print(f"❌ 在 {data_dir} 中没有找到poses.txt或poses.json文件")
            return None, None
        
        print(f"✓ 检测到位姿文件格式: {file_format} ({file_path})")
        
        # 根据格式选择加载方法
        if file_format == "txt":
            return load_dataset_from_txt(
                data_dir=data_dir,
                num_samples=num_samples,
                poses_file="poses.txt",
                clouds_subdir=clouds_subdir
            )
        elif file_format == "json":
            return load_dataset(
                data_dir=data_dir,
                num_samples=num_samples,
                poses_file="poses.json",
                clouds_subdir=clouds_subdir,
                convert_to_rad=True
            )
        else:
            print(f"❌ 不支持的文件格式: {file_format}")
            return None, None
            
    except Exception as e:
        print(f"❌ 自动格式加载数据集失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def parse_joint_states_from_txt(txt_file_path: str) -> List[np.ndarray]:
    """
    从txt文件解析joint_states数据
    
    Args:
        txt_file_path: txt文件路径
        
    Returns:
        joint_states列表，每个元素为6维numpy数组
    """
    joint_states = []
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取joint_states_position数据
        pattern = r"joint_states_position:\s*array\('d',\s*\[(.*?)\]\)"
        matches = re.findall(pattern, content)
        
        for match in matches:
            # 分割数字字符串并转换为浮点数
            numbers_str = match.strip()
            numbers = [float(x.strip()) for x in numbers_str.split(',') if x.strip()]
            
            if len(numbers) == 6:  # 确保是6个关节
                joint_states.append(np.array(numbers))
            else:
                print(f"⚠️  跳过无效的关节状态数据（关节数不为6）: {numbers}")
        
        print(f"✓ 从 {txt_file_path} 成功解析 {len(joint_states)} 个关节状态")
        return joint_states
        
    except Exception as e:
        print(f"❌ 解析txt文件失败 {txt_file_path}: {str(e)}")
        return []


def load_dataset_from_txt(data_dir: str, num_samples: int = None, 
                         poses_file: str = "poses.txt", 
                         clouds_subdir: str = "clouds") -> Tuple[torch.Tensor, List]:
    """
    从txt格式文件加载数据集
    
    Args:
        data_dir: 数据目录路径
        num_samples: 要加载的样本数量，None表示加载全部
        poses_file: 位姿文件名（txt格式）
        clouds_subdir: 点云文件子目录名
        
    Returns:
        (poses, point_clouds) 元组
        poses: torch.Tensor，形状为 (N, 6)
        point_clouds: 点云数据列表
    """
    try:
        # 构建文件路径
        poses_path = os.path.join(data_dir, poses_file)
        clouds_path = os.path.join(data_dir, clouds_subdir)
        
        # 检查文件和目录是否存在
        if not os.path.exists(poses_path):
            print(f"❌ 位姿文件不存在: {poses_path}")
            return None, None
            
        if not os.path.exists(clouds_path):
            print(f"❌ 点云目录不存在: {clouds_path}")
            return None, None
        
        # 解析joint_states数据
        joint_states_list = parse_joint_states_from_txt(poses_path)
        
        if len(joint_states_list) == 0:
            print("❌ 没有有效的关节状态数据")
            return None, None
        
        # 限制样本数量
        if num_samples is not None:
            joint_states_list = joint_states_list[:num_samples]
        
        # 转换为torch张量
        poses = torch.tensor(np.array(joint_states_list), dtype=torch.float64)
        
        # 加载点云数据（复用现有逻辑）
        from data_process.preprocess import load_dataset
        _, point_clouds = load_dataset(
            data_dir=data_dir,
            num_samples=len(joint_states_list),
            clouds_subdir=clouds_subdir,
            convert_to_rad=False  # txt文件中的数据已经是弧度
        )
        
        print(f"✓ 成功从txt格式加载数据集: {len(joint_states_list)} 个位姿，{len(point_clouds) if point_clouds else 0} 个点云")
        return poses, point_clouds
        
    except Exception as e:
        print(f"❌ 从txt格式加载数据集失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def save_observations_to_json(observations: List[Dict[str, Any]], output_file: str) -> bool:
    """
    将观测数据保存为JSON文件
    
    Args:
        observations: 观测数据列表
        output_file: 输出JSON文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 转换数据格式以便JSON序列化
        json_data = {
            "observations": []
        }
        
        for i, obs in enumerate(observations):
            # 计算点云中心
            point_cloud_center = None
            if obs["point_cloud"] is not None:
                try:
                    # 转换为torch张量
                    points_tensor = torch.tensor(obs["point_cloud"], dtype=torch.float64)
                    # 使用球心拟合算法计算中心
                    center_tensor = fit_sphere_center_TLS_A(points_tensor)
                    point_cloud_center = center_tensor.detach().cpu().numpy().tolist()
                except Exception as e:
                    print(f"⚠️  观测 {i} 点云中心计算失败: {str(e)}")
                    point_cloud_center = None
            
            obs_data = {
                "id": i,
                "joint_state": obs["joint_state"].tolist() if obs["joint_state"] is not None else None,
                "point_cloud_center": point_cloud_center  # 保存点云中心坐标而不是完整点云
            }
            json_data["observations"].append(obs_data)
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 成功保存 {len(observations)} 个观测数据到: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 保存观测数据失败: {str(e)}")
        return False


def process_dataset_folder(input_folder: str, output_folder: str, force_overwrite: bool = False) -> bool:
    """
    处理单个数据集文件夹
    
    Args:
        input_folder: 输入数据文件夹路径
        output_folder: 输出数据文件夹路径
        force_overwrite: 是否强制重新处理已存在的文件
        
    Returns:
        是否处理成功
    """
    try:
        print(f"\n=== 处理数据集: {input_folder} ===")
        
        # 检查输出文件是否已存在
        output_file = os.path.join(output_folder, "observations.json")
        if os.path.exists(output_file) and not force_overwrite:
            print(f"⏭️  输出文件已存在，跳过处理: {output_file}")
            return True
        elif os.path.exists(output_file) and force_overwrite:
            print(f"🔄 输出文件已存在，但强制重新处理: {output_file}")
        
        # 检查输入文件夹是否存在
        if not os.path.exists(input_folder):
            print(f"❌ 输入文件夹不存在: {input_folder}")
            return False
        
        # 自动检测位姿文件格式
        file_format, file_path = detect_poses_file_format(input_folder)
        
        if file_format == "none":
            print(f"❌ 在 {input_folder} 中没有找到poses.txt或poses.json文件")
            return False
        
        print(f"✓ 检测到位姿文件格式: {file_format} ({os.path.basename(file_path)})")
        
        # 检查clouds文件夹是否存在
        clouds_folder = os.path.join(input_folder, "clouds")
        if not os.path.exists(clouds_folder):
            print(f"❌ 点云文件夹不存在: {clouds_folder}")
            return False
        
        # 计算点云文件数量
        ply_files = [f for f in os.listdir(clouds_folder) if f.endswith('.ply')]
        num_clouds = len(ply_files)
        print(f"发现 {num_clouds} 个点云文件")
        
        if num_clouds == 0:
            print("❌ 没有找到点云文件")
            return False
        
        # 使用自动格式检测加载数据集
        poses, point_clouds = load_dataset_auto_format(
            data_dir=input_folder,
            num_samples=num_clouds,
            clouds_subdir="clouds"
        )
        
        if poses is None or point_clouds is None:
            print("❌ 数据集加载失败")
            return False
        
        # 创建观测数据
        observations = create_observations(poses, point_clouds)
        
        if len(observations) == 0:
            print("❌ 没有有效的观测数据")
            return False
        
        # 生成输出文件路径
        output_file = os.path.join(output_folder, "observations.json")
        
        # 保存观测数据
        success = save_observations_to_json(observations, output_file)
        
        if success:
            print(f"✓ 数据集处理完成 ({file_format}格式): {len(observations)} 个观测数据")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ 处理数据集失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_all_datasets(origin_data_dir: str = None, output_base_dir: str = None, force_overwrite: bool = False) -> Dict[str, bool]:
    """
    数据预处理主函数 - 批量处理所有数据集
    
    Args:
        origin_data_dir: 原始数据根目录，默认为当前目录下的origin_data
        output_base_dir: 输出数据根目录，默认为当前目录下的data
        force_overwrite: 是否强制重新处理已存在的文件
        
    Returns:
        处理结果字典，键为数据集路径，值为是否处理成功
    """
    # 设置默认路径
    if origin_data_dir is None:
        origin_data_dir = os.path.join(os.getcwd(), "origin_data")
    
    if output_base_dir is None:
        output_base_dir = os.path.join(os.getcwd(), "data")
    
    print("=== 数据预处理主函数 ===")
    print(f"原始数据目录: {origin_data_dir}")
    print(f"输出数据目录: {output_base_dir}")
    
    # 检查原始数据目录是否存在
    if not os.path.exists(origin_data_dir):
        print(f"❌ 原始数据目录不存在: {origin_data_dir}")
        return {}
    
    # 定义要处理的数据集路径映射
    dataset_mapping = {
        # 手眼标定数据
        "hand_eye_data/pose0_test1": "hand_eye_data/pose0_test1",
        "hand_eye_data/pose0_test2": "hand_eye_data/pose0_test2",
        "hand_eye_data/pose2_test1": "hand_eye_data/pose2_test1",
        "hand_eye_data/ABB": "hand_eye_data/ABB",
        "hand_eye_data/ABB_2": "hand_eye_data/ABB_2",
        "hand_eye_data/ABB_3": "hand_eye_data/ABB_3",
        "hand_eye_data/ABB_4": "hand_eye_data/ABB_4",
        "hand_eye_data/ABB_5": "hand_eye_data/ABB_5",
        
        
        # 主要标定数据
        "main_data/pose0": "main_data/pose0",
        "main_data/pose2": "main_data/pose2",
        "main_data/pose3": "main_data/pose3",
        "main_data/pose4": "main_data/pose4",
        "main_data/ABB_1": "main_data/ABB_1",
        "main_data/ABB_2": "main_data/ABB_2",
        
        # 测试数据
        "test_data/pose0": "test_data/pose0",
        "test_data/pose1": "test_data/pose1"
    }
    
    # 处理结果
    results = {}
    successful_count = 0
    total_count = len(dataset_mapping)
    
    # 批量处理每个数据集
    for input_path, output_path in dataset_mapping.items():
        input_folder = os.path.join(origin_data_dir, input_path)
        output_folder = os.path.join(output_base_dir, output_path)
        
        # 处理数据集
        success = process_dataset_folder(input_folder, output_folder, force_overwrite)
        results[input_path] = success
        
        if success:
            successful_count += 1
    
    # 打印处理总结
    print(f"\n=== 数据预处理完成 ===")
    print(f"总数据集数量: {total_count}")
    print(f"成功处理: {successful_count}")
    print(f"失败数量: {total_count - successful_count}")
    
    # 详细结果
    print(f"\n=== 详细处理结果 ===")
    for dataset_path, success in results.items():
        status = "✓ 成功" if success else "❌ 失败"
        print(f"{status}: {dataset_path}")
    
    return results


def preprocess_single_dataset(dataset_path: str, output_path: str = None, force_overwrite: bool = False) -> bool:
    """
    处理单个指定的数据集
    
    Args:
        dataset_path: 数据集路径（相对于origin_data或绝对路径）
        output_path: 输出路径，如果为None则自动生成
        force_overwrite: 是否强制重新处理已存在的文件
        
    Returns:
        是否处理成功
    """
    # 如果是相对路径，则添加origin_data前缀
    if not os.path.isabs(dataset_path):
        origin_data_dir = os.path.join(os.getcwd(), "origin_data")
        input_folder = os.path.join(origin_data_dir, dataset_path)
    else:
        input_folder = dataset_path
    
    # 自动生成输出路径
    if output_path is None:
        data_dir = os.path.join(os.getcwd(), "data")
        if not os.path.isabs(dataset_path):
            output_folder = os.path.join(data_dir, dataset_path)
        else:
            # 从绝对路径提取相对部分
            rel_path = os.path.relpath(dataset_path, os.path.join(os.getcwd(), "origin_data"))
            output_folder = os.path.join(data_dir, rel_path)
    else:
        output_folder = output_path
    
    print(f"处理单个数据集: {input_folder} -> {output_folder}")
    return process_dataset_folder(input_folder, output_folder, force_overwrite)


# 主程序入口
if __name__ == "__main__":
    # 示例1: 处理所有数据集（跳过已存在的文件）
    print("=== 示例1: 批量处理所有数据集（跳过已存在） ===")
    results = preprocess_all_datasets()
    print("处理结果:", results)