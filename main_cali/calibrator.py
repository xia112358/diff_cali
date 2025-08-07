import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from robot_model.unified_robot import BaseRobot
from typing import List, Dict
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import random


class CalibrationDataset(Dataset):
    """
    校准数据集类，用于PyTorch DataLoader
    """
    def __init__(self, joint_positions, sphere_centers_sensor):
        self.joint_positions = joint_positions
        self.sphere_centers_sensor = sphere_centers_sensor
    
    def __len__(self):
        return len(self.joint_positions)
    
    def __getitem__(self, idx):
        return idx



class RobotCalibrator:
    """
    基于位置方差的机器人运动学参数校准类
    
    使用torch的自动微分进行参数优化,支持DH参数和beta参数的校准
    使用预处理的JSON观测数据格式,包含预计算的球心位置
    目标函数:最小化所有观测点位置的方差
    
    支持新的统一机器人接口(BaseRobot及其子类)
    
    性能优化功能:
    - 自动选择计算方法:批量、向量化或并行
    - 多线程并行计算支持(大数据集时)
    - 批量张量计算优化
    """
    
    def __init__(self, robot: BaseRobot, calibration_params: List[str] = None, 
         excluded_params: Dict[int, List[str]] = None, target_precision: float = 1e-6):
        """
        初始化校准器
        
        :param robot: 机器人模型
        :param calibration_params: 要校准的参数名称列表
        :param excluded_params: 要排除的参数,格式:{关节索引: [参数名列表]}
        :param target_precision: 目标精度(方差阈值)
        """
        self.robot = robot
        
        if calibration_params is None:
            # 默认校准参数包括手眼参数
            self.calibration_params = ['a', 'alpha', 'd', 'beta', 'offset', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        else:
            self.calibration_params = calibration_params
        
        # 默认不排除任何参数
        if excluded_params is None:
            self.excluded_params = {}  # 空字典,不排除任何参数
        else:
            self.excluded_params = excluded_params
        
        self.target_precision = target_precision
        self.joint_positions = []
        self.sphere_centers_sensor = []
        self.observations = []
        self.calibration_results = None
        self.optimization_history = []
        self.learning_rate = 0.01
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.sphere_centers_base_before = []
        self.sphere_centers_base_after = []
    
    def set_observations(self, observations: List[Dict], max_observations: int = None):
        """
        设置观测数据 - 使用新的JSON数据格式

        :param observations: 观测数据列表, 每个元素包含 joint_state 和 point_cloud_center
        :param max_observations: 最大读取观测数据个数，None表示读取全部
        """
        # 检查输入格式
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list of dictionaries.")

        # 限制观测数据数量
        if max_observations is not None and max_observations > 0:
            observations = observations[:max_observations]

        # 清空现有数据
        self.joint_positions = []
        self.sphere_centers_sensor = []  # 用于保存点云中心
        self.observations = observations  # 保存原始观测数据

        # 提取观测数据
        for i, obs in enumerate(observations):
            # 检查必需字段
            if "joint_state" not in obs:
                raise ValueError(f"Observation {i} must contain 'joint_state'.")
            if "point_cloud_center" not in obs:
                raise ValueError(f"Observation {i} must contain 'point_cloud_center'.")

            # 保存关节状态 - 转换为 torch 张量
            joint_state = obs["joint_state"]
            if not isinstance(joint_state, torch.Tensor):
                joint_state = torch.tensor(joint_state, dtype=torch.float64)
            else:
                joint_state = joint_state.detach().clone()  # 移除梯度并克隆
            self.joint_positions.append(joint_state)

            # 处理点云中心数据
            point_cloud_center = obs["point_cloud_center"]
            if point_cloud_center is not None:
                if not isinstance(point_cloud_center, torch.Tensor):
                    center = torch.tensor(point_cloud_center, dtype=torch.float64)
                else:
                    center = point_cloud_center.detach().clone()
                self.sphere_centers_sensor.append(center)
            else:
                raise ValueError(f"Observation {i} has null point_cloud_center.")

        print(f"Loaded {len(self.joint_positions)} observations with precomputed centers")
    
    def load_observations_from_json(self, json_file_path: str, max_observations: int = None):
        """
        从JSON文件加载观测数据
        
        :param json_file_path: observations.json文件路径
        :param max_observations: 最大读取观测数据个数，None表示读取全部
        :return: 加载是否成功
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'observations' not in data:
                raise ValueError("JSON file must contain 'observations' field")
            
            observations = data['observations']
            total_observations = len(observations)
            
            self.set_observations(observations, max_observations)
            
            if max_observations is not None and max_observations < total_observations:
                print(f"✓ 成功从 {json_file_path} 加载 {len(self.observations)} 个观测数据 (共 {total_observations} 个)")
            else:
                print(f"✓ 成功从 {json_file_path} 加载 {len(self.observations)} 个观测数据")
            return True
            
        except Exception as e:
            print(f"❌ 从JSON文件加载观测数据失败: {str(e)}")
            return False
    
    # 已移除mini-batch相关的DataLoader方法
    
    # 已移除未使用的随机采样方法
    
    def _compute_sphere_centers_in_base(self) -> List[torch.Tensor]:
        """
        计算所有观测中球心在基坐标系下的位置
        
        :return: 球心位置列表
        """
        sphere_centers_base = []
        
        for i in range(len(self.joint_positions)):
            q = self.joint_positions[i]
            sphere_center_sensor = self.sphere_centers_sensor[i]
            
            # 计算前向运动学:传感器到基坐标系的变换
            T_sensor_to_base = self.robot.fkine(q)
            
            # 将球心从传感器坐标系转换到基坐标系
            sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
                torch.tensor(sphere_center_sensor, dtype=torch.float64) if not isinstance(sphere_center_sensor, torch.Tensor) else sphere_center_sensor, 
                torch.tensor([1.0], dtype=torch.float64)
            ])
            
            # 取前3个元素(去掉齐次坐标)
            sphere_center_base_calculated = sphere_center_base_calculated[:3]
            sphere_centers_base.append(sphere_center_base_calculated)
        
        return sphere_centers_base
    
    def _compute_variance_error(self, batch_indices=None) -> torch.Tensor:
        """
        计算观测点位置的方差 - 自动选择最优计算方法
        支持batch输入和全量计算
        
        :param batch_indices: 可选，指定要计算的观测点索引列表
        :return: 方差值
        """
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        # 如果指定了batch_indices，只使用batch中的数据
        if batch_indices is not None:
            joint_positions = [self.joint_positions[i] for i in batch_indices]
            sphere_centers = [self.sphere_centers_sensor[i] for i in batch_indices]
            n_observations = len(joint_positions)
        else:
            joint_positions = self.joint_positions
            sphere_centers = self.sphere_centers_sensor
            n_observations = len(joint_positions)
        
        # 自动选择计算方法基于数据量
        if n_observations > 100:
            # 大数据量：尝试使用并行计算
            try:
                return self._compute_variance_error_parallel_optimized(joint_positions, sphere_centers)
            except Exception as e:
                print(f"Parallel computation failed: {e}, falling back to vectorized")
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
        elif n_observations > 20:
            # 中等数据量：使用向量化计算
            try:
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
            except Exception as e:
                print(f"Vectorized computation failed: {e}, falling back to batch")
                return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)
        else:
            # 小数据量：使用批量计算
            return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)
    
    def _compute_variance_error_batch_optimized(self, joint_positions, sphere_centers) -> torch.Tensor:
        """
        优化的批量计算方差误差
        
        :param joint_positions: 关节位置列表
        :param sphere_centers: 球心位置列表
        :return: 方差值
        """
        n_observations = len(joint_positions)
        
        # 将所有关节位置和球心转换为张量批量处理
        joint_positions_batch = torch.stack(joint_positions)  # [n_observations, n_joints]
        sphere_centers_batch = torch.stack(sphere_centers)  # [n_observations, 3]
        
        # 批量计算所有位置
        calculated_positions = []
        
        for i in range(n_observations):
            # 计算前向运动学:传感器到基坐标系的变换
            T_sensor_to_base = self.robot.fkine(joint_positions_batch[i])
            
            # 将球心从传感器坐标系转换到基坐标系
            sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
                sphere_centers_batch[i], 
                torch.tensor([1.0], dtype=torch.float64)
            ])
            
            # 取前3个元素(去掉齐次坐标)
            calculated_positions.append(sphere_center_base_calculated[:3])
        
        # 将所有位置堆叠成张量
        positions_tensor = torch.stack(calculated_positions)  # [n_observations, 3]
        
        # 计算每个维度的方差,然后求和
        variance = torch.var(positions_tensor, dim=0).sum()  # 计算x,y,z三个维度的方差之和
        
        return variance
    
    def _compute_variance_error_vectorized_optimized(self, joint_positions, sphere_centers) -> torch.Tensor:
        """
        优化的向量化计算方差误差
        
        :param joint_positions: 关节位置列表
        :param sphere_centers: 球心位置列表
        :return: 方差值
        """
        n_observations = len(joint_positions)
        
        # 尝试批量计算所有前向运动学(如果robot支持)
        try:
            # 将所有关节位置和球心位置转换为批量张量
            joint_positions_batch = torch.stack(joint_positions)  # [n_observations, n_joints]
            sphere_centers_batch = torch.stack(sphere_centers)  # [n_observations, 3]

            # 如果机器人支持批量前向运动学,使用它
            if hasattr(self.robot, 'fkine_batch'):
                T_batch = self.robot.fkine_batch(joint_positions_batch)
                # 批量变换所有球心位置
                ones = torch.ones(len(sphere_centers_batch), 1, dtype=torch.float64)
                homogeneous_centers = torch.cat([sphere_centers_batch, ones], dim=1)  # [n_obs, 4]

                # 批量矩阵乘法
                transformed_centers = torch.bmm(T_batch, homogeneous_centers.unsqueeze(-1)).squeeze(-1)
                calculated_positions = transformed_centers[:, :3]  # 取前3个元素
            else:
                # 回退到批量计算
                return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)

            # 计算方差
            variance = torch.var(calculated_positions, dim=0).sum()
            return variance

        except Exception as e:
            # 如果向量化失败,回退到批量计算
            return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)
    
    def _compute_variance_error_parallel_optimized(self, joint_positions, sphere_centers) -> torch.Tensor:
        """
        优化的并行计算方差误差
        
        :param joint_positions: 关节位置列表
        :param sphere_centers: 球心位置列表
        :return: 方差值
        """
        n_observations = len(joint_positions)
        
        if n_observations > 200:  # 只在样本数量足够大时使用并行
            try:
                # 使用线程池而不是进程池,减少启动开销
                max_workers = min(4, max(2, n_observations // 50))
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 并行计算所有位置
                    futures = []
                    for joint_pos, sphere_center in zip(joint_positions, sphere_centers):
                        future = executor.submit(self._compute_single_position, (joint_pos, sphere_center))
                        futures.append(future)
                    
                    # 收集结果
                    calculated_positions = [future.result() for future in futures]
                
                # 将结果转换为张量并计算方差
                positions_tensor = torch.stack(calculated_positions)  # [n_observations, 3]
                variance = torch.var(positions_tensor, dim=0).sum()  # 计算x,y,z三个维度的方差之和
                return variance
            except Exception as e:
                # 并行计算失败,回退到向量化计算
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
        else:
            # 样本数不够大,直接使用向量化计算
            return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
    
    def _get_calibration_parameters(self) -> List[torch.Tensor]:
        """
        获取需要校准的参数
        使用可配置的排除规则
    
        :return: 参数列表
        """
        params = []
        
        # 获取关节参数 (适配新的统一接口)
        if hasattr(self.robot, 'links'):
            # MDH机器人模型
            for i, link in enumerate(self.robot.links):
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        # 检查是否在排除列表中
                        if i in self.excluded_params and param_name in self.excluded_params[i]:
                            continue  # 跳过被排除的参数
                        
                        param = getattr(link, param_name)
                        param.requires_grad = True
                        params.append(param)
        
        # 获取手眼参数 (新的统一接口中直接作为机器人属性)
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                # 直接访问机器人的手眼参数 (不再是私有属性)
                if hasattr(self.robot, param_name):
                    param = getattr(self.robot, param_name)
                    param.requires_grad = True
                    params.append(param)
        
        return params
    
    # 高精度参数校准方法
    def calibrate(self, learning_rate: float = None, max_iterations: int = None, 
                  tolerance: float = None, verbose: bool = True) -> Dict:
        """
        执行高精度参数校准
        
        :param learning_rate: 学习率 (默认使用高精度设置)
        :param max_iterations: 最大迭代次数 (默认使用高精度设置)
        :param tolerance: 收敛容差 (默认使用高精度设置)
        :param verbose: 是否打印详细信息
        :param use_mini_batch: 是否使用mini-batch训练
        :param batch_size: mini-batch大小
        :return: 校准结果字典
        """
        if len(self.joint_positions) == 0:
            raise ValueError("No calibration data, please set observations first")
        
        # 使用高精度默认参数
        lr = learning_rate if learning_rate is not None else self.learning_rate
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        tol = tolerance if tolerance is not None else self.tolerance
        
        # 记录初始参数值
        initial_params = {}
        # 记录关节参数 (适配新的统一接口)
        if hasattr(self.robot, 'links'):
            # MDH机器人模型
            for i, link in enumerate(self.robot.links):
                initial_params[f'link_{i}'] = {}
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        initial_params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        # 记录手眼参数 (新的统一接口中直接作为机器人属性)
        initial_params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                # 直接访问机器人的手眼参数 (不再是私有属性)
                if hasattr(self.robot, param_name):
                    initial_params['hand_eye'][param_name] = getattr(self.robot, param_name).item()
        
        # 记录初始误差
        initial_error = self._compute_variance_error().item()
        
        # 记录校准前的球心位置
        self.sphere_centers_base_before = self._compute_sphere_centers_in_base()
        
        # 获取需要校准的参数
        params = self._get_calibration_parameters()
        
        if len(params) == 0:
            raise ValueError("No parameters found for calibration")
        
        # 清空优化历史
        self.optimization_history = []
        
        # 使用AdamW优化器
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=50
        )
        
        if verbose:
            print(f"Initial Variance Error: {initial_error:.6f}")
            print(f"Target precision: {self.target_precision:.10f}")
            print(f"Using full-batch training")
            
            # 显示自动选择的计算方法
            n_obs = len(self.joint_positions)
            if n_obs > 100:
                print(f"Data size: {n_obs}, using parallel computation for acceleration")
            elif n_obs > 20:
                print(f"Data size: {n_obs}, using vectorized computation for acceleration")
            else:
                print(f"Data size: {n_obs}, using batch computation")
        
        # 优化循环
        prev_loss = float('inf')
        for iteration in range(max_iter):
            optimizer.zero_grad()
            variance_error = self._compute_variance_error()
            current_loss = variance_error.item()
            full_loss = current_loss
            # 在更新之前进行判断
            if current_loss < self.target_precision:
                if verbose:
                    print(f"Target precision achieved at iteration {iteration}")
                break
            if abs(prev_loss - current_loss) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            variance_error.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step(full_loss)
            self.optimization_history.append(full_loss)
            if verbose and iteration % 1000 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iteration {iteration}: Variance Error = {current_loss:.16f}, LR = {current_lr:.12f}")
            prev_loss = current_loss
        
        # 记录校准后的球心位置
        self.sphere_centers_base_after = self._compute_sphere_centers_in_base()
        
        # 记录最终参数值
        final_params = {}
        # 记录关节参数 (适配新的统一接口)
        if hasattr(self.robot, 'links'):
            # MDH机器人模型
            for i, link in enumerate(self.robot.links):
                final_params[f'link_{i}'] = {}
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        final_params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        # 记录手眼参数 (新的统一接口中直接作为机器人属性)
        final_params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                # 直接访问机器人的手眼参数 (不再是私有属性)
                if hasattr(self.robot, param_name):
                    final_params['hand_eye'][param_name] = getattr(self.robot, param_name).item()
        
        # 计算最终误差
        final_variance_error = self._compute_variance_error().item()
        
        # 判断是否达到目标精度
        precision_achieved = final_variance_error < self.target_precision
        
        # 保存结果
        self.calibration_results = {
            'initial_params': initial_params,
            'final_params': final_params,
            'initial_variance_error': initial_error,
            'final_variance_error': final_variance_error,
            'target_precision': self.target_precision,
            'precision_achieved': precision_achieved,
            'iterations': len(self.optimization_history),
            'converged': iteration < max_iter - 1,
            'calibration_time': datetime.now().isoformat(),
            'optimization_history': self.optimization_history,  # 添加优化历史
            'sphere_centers_base_before': [center.tolist() for center in self.sphere_centers_base_before],  # 添加校准前的球心位置
            'sphere_centers_base_after': [center.tolist() for center in self.sphere_centers_base_after],  # 添加校准后的球心位置
        }
        
        if verbose:
            print(f"\nCalibration completed!")
            print(f"Initial Variance Error: {initial_error:.6f}")
            print(f"Final Variance Error: {final_variance_error:.16f}")
            print(f"Target precision: {self.target_precision:.16f}")
            print(f"Precision achieved: {'Yes' if precision_achieved else 'No'}")
            
            if initial_error > 0:
                improvement = ((initial_error - final_variance_error) / initial_error * 100)
                print(f"Error Improvement: {improvement:.10f}%")
    
        return self.calibration_results
    
    
    
    # 已移除未使用的benchmark_computation_methods方法
    
    def save_results(self, filename: str):
        """
        保存校准结果到result文件夹
        
        :param filename: 文件名
        """
        if self.calibration_results is None:
            raise ValueError("No calibration results available. Please run calibrate() first.")
        
        import os
        # 确保result文件夹存在
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 将文件保存到result文件夹
        filepath = os.path.join(result_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.calibration_results, f, indent=2)
        
        print(f"Calibration results saved to {filepath}")
    
    
    
    def _compute_single_position(self, joint_sphere_pair):
        """
        计算单个观测的球心位置(用于并行计算)
        
        :param joint_sphere_pair: (joint_positions, sphere_center_sensor)的元组
        :return: 计算得到的球心位置
        """
        joint_pos, sphere_center_sensor = joint_sphere_pair
        
        # 计算前向运动学:传感器到基坐标系的变换
        T_sensor_to_base = self.robot.fkine(joint_pos)
        
        # 将球心从传感器坐标系转换到基坐标系
        sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
            sphere_center_sensor, 
            torch.tensor([1.0], dtype=torch.float64)
        ])
        
        # 取前3个元素(去掉齐次坐标)
        return sphere_center_base_calculated[:3]


