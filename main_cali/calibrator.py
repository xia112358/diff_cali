import torch
import torch.optim as optim
from .DHRobotWithBeta_torch import DHRobotWithBeta
from typing import List, Dict
import json
from datetime import datetime
from .cloud import fit_sphere_center_TLS_A
from concurrent.futures import ThreadPoolExecutor



class RobotCalibrator:
    """
    基于位置方差的机器人运动学参数校准类
    
    使用torch的自动微分进行参数优化,支持DH参数和beta参数的校准
    兼容sphere-based观测数据格式
    目标函数:最小化所有观测点位置的方差
    
    性能优化功能:
    - 自动选择计算方法:串行、批量或并行
    - 多进程并行计算支持(样本数>50时)
    - 批量张量计算(样本数10-50时)
    """
    
    def __init__(self, robot: DHRobotWithBeta, calibration_params: List[str] = None, 
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
        
        # 校准结果
        self.calibration_results = None
        self.optimization_history = []
        
        # 高精度优化器参数
        self.learning_rate = 0.01         # 提高学习率
        self.max_iterations = 1000        # 减少最大迭代次数
        self.tolerance = 1e-6             # 适当放宽收敛容差
        self.target_precision = target_precision      # 目标精度 
        
        # 学习率调度器参数
        self.use_scheduler = True
        self.scheduler_patience = 50
        
        # 存储校准前后的计算结果用于可视化
        self.sphere_centers_base_before = []
        self.sphere_centers_base_after = []
    
    def set_observations(self, observations: List[Dict]):
        """
        设置观测数据 - 提取点云中心并存储

        :param observations: 观测数据列表, 每个元素包含 joint_state 和 point_cloud
        """
        # 检查输入格式
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list of dictionaries with 'joint_state' and 'point_cloud' keys.")

        # 清空现有数据
        self.joint_positions = []
        self.sphere_centers_sensor = []  # 用于保存点云中心

        # 提取观测数据
        for i, obs in enumerate(observations):
            if "joint_state" not in obs or "point_cloud" not in obs:
                raise ValueError(f"Observation {i} must contain 'joint_state' and 'point_cloud'.")

            # 保存关节状态 - 转换为 torch 张量
            joint_state = obs["joint_state"]
            if not isinstance(joint_state, torch.Tensor):
                joint_state = torch.tensor(joint_state, dtype=torch.float64)
            else:
                joint_state = joint_state.detach().clone()  # 移除梯度并克隆
            self.joint_positions.append(joint_state)

            # 提取点云中心
            point_cloud = torch.stack(obs["point_cloud"]) if isinstance(obs["point_cloud"], list) else obs["point_cloud"]
            # 确保 point_cloud 是 torch 张量，因为 fit_sphere_center_TLS_A 期望张量输入
            if not isinstance(point_cloud, torch.Tensor):
                point_cloud = torch.tensor(point_cloud, dtype=torch.float64)
            else:
                point_cloud = point_cloud.detach()  # 移除梯度但保持为张量
            center = fit_sphere_center_TLS_A(point_cloud)  # 使用高精度拟合球心位置
            self.sphere_centers_sensor.append(center)

        print(f"Loaded {len(self.joint_positions)} observations with computed centers")
    
    def _fit_sphere_center(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        从点云拟合球心位置(高精度版本)
        
        :param point_cloud: 点云数据 (N x 3) torch 张量
        :return: 球心位置
        """
        # 使用高精度拟合球心
        center = fit_sphere_center_TLS_A(point_cloud) 
        
        return center
    
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
    
    def _compute_variance_error(self) -> torch.Tensor:
        """
        计算所有观测点位置的方差
        自动选择最优的计算方法:并行、批量或串行
        
        :return: 方差值
        """
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        n_observations = len(self.joint_positions)
        
        # 根据样本数量自动选择计算方法
        # 优先尝试向量化计算,然后是批量计算
        if n_observations > 1000:  # 非常大的样本才使用并行
            return self._compute_variance_error_parallel()
        elif n_observations > 5:  # 尝试向量化计算
            try:
                return self._compute_variance_error_vectorized()
            except:
                # 向量化失败,使用批量计算
                return self._compute_variance_error_batch()
        # 否则使用串行计算(当前实现)
        
        # 串行计算实现
        calculated_positions = []
        
        for i in range(n_observations):
            # 获取关节角度和传感器坐标系中的球心
            q = self.joint_positions[i]
            sphere_center_sensor = self.sphere_centers_sensor[i]
            
            # 计算前向运动学:传感器到基坐标系的变换
            T_sensor_to_base = self.robot.fkine(q)
            
            # 将球心从传感器坐标系转换到基坐标系
            sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
                sphere_center_sensor, 
                torch.tensor([1.0], dtype=torch.float64)
            ])
            
            # 取前3个元素(去掉齐次坐标)
            sphere_center_base_calculated = sphere_center_base_calculated[:3]
            calculated_positions.append(sphere_center_base_calculated)
        
        # 将所有位置堆叠成张量
        positions_tensor = torch.stack(calculated_positions)  # [n_observations, 3]
        
        # 计算每个维度的方差,然后求和
        variance = torch.var(positions_tensor, dim=0).sum()  # 计算x,y,z三个维度的方差之和
        
        return variance
    
    def _compute_variance_error_serial(self) -> torch.Tensor:
        """
        串行计算方差误差(用于性能比较)
        
        :return: 方差值
        """
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        n_observations = len(self.joint_positions)
        
        # 串行计算所有位置
        calculated_positions = []
        
        for i in range(n_observations):
            # 获取关节角度和传感器坐标系中的球心
            q = self.joint_positions[i]
            sphere_center_sensor = self.sphere_centers_sensor[i]
            
            # 计算前向运动学:传感器到基坐标系的变换
            T_sensor_to_base = self.robot.fkine(q)
            
            # 将球心从传感器坐标系转换到基坐标系
            sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
                sphere_center_sensor, 
                torch.tensor([1.0], dtype=torch.float64)
            ])
            
            # 取前3个元素(去掉齐次坐标)
            calculated_positions.append(sphere_center_base_calculated[:3])
        
        # 将所有位置堆叠成张量
        positions_tensor = torch.stack(calculated_positions)  # [n_observations, 3]
        
        # 计算每个维度的方差,然后求和
        variance = torch.var(positions_tensor, dim=0).sum()
        
        return variance
    
    def _compute_variance_error_vectorized(self) -> torch.Tensor:
        """
        使用完全向量化计算提高方差计算效率(实验性)
        
        :return: 方差值
        """
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        # 尝试批量计算所有前向运动学(如果robot支持)
        try:
            # 将所有关节位置和球心位置转换为批量张量
            joint_positions_batch = torch.stack([torch.tensor(jp, dtype=torch.float64) if not isinstance(jp, torch.Tensor) else jp for jp in self.joint_positions])  # [n_observations, n_joints]
            sphere_centers_batch = torch.stack([torch.tensor(sc, dtype=torch.float64) if not isinstance(sc, torch.Tensor) else sc for sc in self.sphere_centers_sensor])  # [n_observations, 3]

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
                return self._compute_variance_error_batch()

            # 计算方差
            variance = torch.var(calculated_positions, dim=0).sum()
            return variance

        except Exception as e:
            # 如果向量化失败,回退到批量计算
            return self._compute_variance_error_batch()
    
    def _compute_variance_error_batch(self) -> torch.Tensor:
        """
        使用批量计算提高方差计算效率
        
        :return: 方差值
        """
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        n_observations = len(self.joint_positions)
        
        # 将所有关节位置转换为张量批量处理
        joint_positions_batch = torch.stack(self.joint_positions)  # [n_observations, n_joints]
        sphere_centers_batch = torch.stack(self.sphere_centers_sensor)  # [n_observations, 3]
        
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
    
    def _get_calibration_parameters(self) -> List[torch.Tensor]:
        """
        获取需要校准的参数
        使用可配置的排除规则
    
        :return: 参数列表
        """
        params = []
        
        # 获取关节参数
        for i, link in enumerate(self.robot.links):
            for param_name in self.calibration_params:
                if hasattr(link, param_name):
                    # 检查是否在排除列表中
                    if i in self.excluded_params and param_name in self.excluded_params[i]:
                        continue  # 跳过被排除的参数
                    
                    param = getattr(link, param_name)
                    param.requires_grad = True
                    params.append(param)
        
        # 获取手眼参数(仅当机器人不使用固定tool变换时)
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                # 检查机器人是否支持可校准的手眼参数
                if hasattr(self.robot, '_use_custom_tool') and self.robot._use_custom_tool:
                    # 机器人使用固定tool变换,跳过手眼参数
                    continue
                    
                # 访问机器人的私有手眼参数
                private_param_name = f'_{param_name}'
                if hasattr(self.robot, private_param_name):
                    param = getattr(self.robot, private_param_name)
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
        # 记录关节参数
        for i, link in enumerate(self.robot.links):
            initial_params[f'link_{i}'] = {}
            for param_name in self.calibration_params:
                if hasattr(link, param_name):
                    initial_params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        # 记录手眼参数(仅当机器人不使用固定tool变换时)
        initial_params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                # 检查机器人是否支持可校准的手眼参数
                if hasattr(self.robot, '_use_custom_tool') and self.robot._use_custom_tool:
                    # 机器人使用固定tool变换,跳过手眼参数
                    continue
                    
                # 访问机器人的私有手眼参数
                private_param_name = f'_{param_name}'
                if hasattr(self.robot, private_param_name):
                    initial_params['hand_eye'][param_name] = getattr(self.robot, private_param_name).item()
        
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
        
        # 优化循环
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            variance_error = self._compute_variance_error()
            current_loss = variance_error.item()
            
            # 在更新之前进行判断
            # 检查是否达到目标精度
            if current_loss < self.target_precision:
                if verbose:
                    print(f"Target precision achieved at iteration {iteration}")
                break
            
            # 检查收敛
            if abs(prev_loss - current_loss) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # 执行优化器更新
            variance_error.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            # 更新学习率调度器
            scheduler.step(current_loss)
            
            self.optimization_history.append(current_loss)
            
            # 打印进度
            if verbose and iteration % 1000 == 0:  # 每1000次迭代打印一次
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iteration {iteration}: Variance Error = {current_loss:.16f}, LR = {current_lr:.12f}")
            
            prev_loss = current_loss
        
        # 记录校准后的球心位置
        self.sphere_centers_base_after = self._compute_sphere_centers_in_base()
        
        # 记录最终参数值
        final_params = {}
        # 记录关节参数
        for i, link in enumerate(self.robot.links):
            final_params[f'link_{i}'] = {}
            for param_name in self.calibration_params:
                if hasattr(link, param_name):
                    final_params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        # 记录手眼参数(仅当机器人不使用固定tool变换时)
        final_params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                # 检查机器人是否支持可校准的手眼参数
                if hasattr(self.robot, '_use_custom_tool') and self.robot._use_custom_tool:
                    # 机器人使用固定tool变换,跳过手眼参数
                    continue
                    
                # 访问机器人的私有手眼参数
                private_param_name = f'_{param_name}'
                if hasattr(self.robot, private_param_name):
                    final_params['hand_eye'][param_name] = getattr(self.robot, private_param_name).item()
        
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
    
    def _compute_variance_error_parallel(self):
        """
        使用多线程并行计算方差误差(改用线程池减少开销)
        """
        if len(self.joint_positions) > 500:  # 只在样本数量非常大时使用并行
            try:
                # 使用线程池而不是进程池,减少启动开销
                max_workers = min(4, max(2, len(self.joint_positions) // 100))
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 并行计算所有位置
                    futures = []
                    for joint_pos, sphere_center in zip(self.joint_positions, self.sphere_centers_sensor):
                        future = executor.submit(self._compute_single_position, (joint_pos, sphere_center))
                        futures.append(future)
                    
                    # 收集结果
                    calculated_positions = [future.result() for future in futures]
                
                # 将结果转换为张量并计算方差
                positions_tensor = torch.stack(calculated_positions)  # [n_observations, 3]
                variance = torch.var(positions_tensor, dim=0).sum()  # 计算x,y,z三个维度的方差之和
                return variance
            except Exception as e:
                print(f"Parallel computation failed: {e}, falling back to batch computation")
                return self._compute_variance_error_batch()
        else:
            # 样本数不够大,直接使用批量计算
            return self._compute_variance_error_batch()
    
    def _compute_errors_parallel(self):
        """
        使用多进程并行计算误差(保持向后兼容)
        """
        return self._compute_variance_error_parallel()


