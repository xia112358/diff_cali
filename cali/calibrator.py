import torch
import torch.optim as optim
from cali.mdh_robot import DHRobotWithBeta
from typing import List, Dict
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class RobotCalibrator:
    """基于位置方差的机器人运动学参数校准类，支持自动微分优化和离群值检测。"""
    
    def __init__(self, robot: DHRobotWithBeta, 
                 calibration_params: List[str] = None, 
                 excluded_params: Dict[int, List[str]] = None,
                 # ========== 优化器超参数 ==========
                 learning_rate: float = 0.01,              # 初始学习率
                 weight_decay: float = 1e-5,               # L2正则化系数，防止过拟合
                 max_iterations: int = 1000,               # 最大迭代次数
                 gradient_clip_norm: float = 1.0,          # 梯度裁剪阈值，防止梯度爆炸
                 # ========== 学习率调度器超参数 ==========
                 scheduler_factor: float = 0.7,            # 学习率衰减因子
                 scheduler_patience: int = 50,             # 学习率衰减耐心值
                 # ========== 收敛和精度控制 ==========
                 target_precision: float = 1e-6,          # 目标精度（方差误差阈值）
                 convergence_tolerance: float = 1e-6,      # 连续两次loss差值小于此值时判断收敛
                 # ========== 离群值检测超参数 ==========
                 enable_outlier_detection: bool = False,   # 是否启用离群值检测
                 outlier_detection_interval: int = 50,     # 离群值检测间隔
                 outlier_threshold: float = 1.5,           # 离群值检测阈值（基于IQR的倍数）
                 max_outlier_ratio: float = 0.3,           # 最大离群值比例
                 max_outliers_per_round: int = 10,         # 每轮最多剔除的离群值数量
                 # ========== 计算优化超参数 ==========
                 parallel_threshold: int = 200,            # 启用并行计算的数据量阈值
                 vectorized_threshold: int = 5,            # 启用向量化计算的数据量阈值
                 max_parallel_workers: int = 4,            # 并行计算最大工作线程数
                 # ========== 日志和输出控制 ==========
                 log_interval: int = 1000,                 # 训练日志输出间隔
                 verbose: bool = True                       # 是否输出详细信息
                 ):
        """
        初始化校准器
        
        Args:
            robot: DH机器人模型
            calibration_params: 需要校准的参数列表
            excluded_params: 排除的参数字典 {关节索引: [参数名列表]}
            learning_rate: 初始学习率
            weight_decay: L2正则化系数
            max_iterations: 最大迭代次数
            gradient_clip_norm: 梯度裁剪阈值
            scheduler_factor: 学习率衰减因子
            scheduler_patience: 学习率衰减耐心值
            target_precision: 目标精度（方差误差阈值）
            convergence_tolerance: 收敛判断阈值
            enable_outlier_detection: 是否启用离群值检测
            outlier_detection_interval: 离群值检测间隔
            outlier_threshold: 离群值检测阈值
            max_outlier_ratio: 最大离群值比例
            max_outliers_per_round: 每轮最多剔除的离群值数量
            parallel_threshold: 启用并行计算的数据量阈值
            vectorized_threshold: 启用向量化计算的数据量阈值
            max_parallel_workers: 并行计算最大工作线程数
            log_interval: 训练日志输出间隔
            verbose: 是否输出详细信息
        """
        self.robot = robot
        
        # 设置校准参数
        if calibration_params is None:
            self.calibration_params = ['a', 'alpha', 'd', 'beta', 'offset', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        else:
            self.calibration_params = calibration_params
        
        # 设置排除参数
        self.excluded_params = excluded_params if excluded_params is not None else {}
        
        # ========== 优化器超参数 ==========
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_iterations = max_iterations
        self.gradient_clip_norm = gradient_clip_norm
        
        # ========== 学习率调度器超参数 ==========
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        
        # ========== 收敛和精度控制 ==========
        self.target_precision = target_precision
        self.convergence_tolerance = convergence_tolerance
        
        # ========== 离群值检测超参数 ==========
        self.enable_outlier_detection = enable_outlier_detection
        self.outlier_detection_interval = outlier_detection_interval
        self.outlier_threshold = outlier_threshold
        self.max_outlier_ratio = max_outlier_ratio
        self.max_outliers_per_round = max_outliers_per_round
        
        # ========== 计算优化超参数 ==========
        self.parallel_threshold = parallel_threshold
        self.vectorized_threshold = vectorized_threshold
        self.max_parallel_workers = max_parallel_workers
        
        # ========== 日志和输出控制 ==========
        self.log_interval = log_interval
        self.verbose = verbose
        
        # 数据存储
        self.joint_positions = []
        self.sphere_centers_sensor = []
        self.observations = []
        self.calibration_results = None
        self.optimization_history = []
        self.sphere_centers_base_before = []
        self.sphere_centers_base_after = []
        
        # 离群值检测相关
        self.outlier_indices = set()
    
    def set_observations(self, observations: List[Dict], max_observations: int = None):
        """设置观测数据"""
        if not isinstance(observations, list):
            raise ValueError("Observations must be a list of dictionaries.")

        if max_observations is not None and max_observations > 0:
            observations = observations[:max_observations]

        self.joint_positions = []
        self.sphere_centers_sensor = []
        self.observations = observations

        for i, obs in enumerate(observations):
            if "joint_state" not in obs:
                raise ValueError(f"Observation {i} must contain 'joint_state'.")
            if "point_cloud_center" not in obs:
                raise ValueError(f"Observation {i} must contain 'point_cloud_center'.")

            joint_state = obs["joint_state"]
            if not isinstance(joint_state, torch.Tensor):
                joint_state = torch.tensor(joint_state, dtype=torch.float64)
            else:
                joint_state = joint_state.detach().clone()
            self.joint_positions.append(joint_state)

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
        """从JSON文件加载观测数据"""
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
    
    def _detect_and_remove_outliers(self, verbose: bool = True) -> int:
        """
        检测并剔除离群值观测数据
        
        使用四分位距(IQR)方法检测离群值：
        - 计算所有观测点在基坐标系下的位置
        - 基于距离中心点的距离，使用IQR方法识别离群值
        - 控制每轮剔除数量和总剔除比例
        
        Args:
            verbose: 是否输出详细信息
            
        Returns:
            本轮剔除的离群值数量
        """
        if not self.enable_outlier_detection:
            return 0
        if len(self.joint_positions) == 0:
            return 0
        
        sphere_centers_base = self._compute_sphere_centers_in_base()
        positions_tensor = torch.stack(sphere_centers_base)
        center_position = torch.mean(positions_tensor, dim=0)
        distances = torch.norm(positions_tensor - center_position, dim=1)
        
        # 使用四分位距方法检测离群值
        q1 = torch.quantile(distances, 0.25)
        q3 = torch.quantile(distances, 0.75)
        iqr = q3 - q1
        threshold = q3 + self.outlier_threshold * iqr
        
        # 找出新的离群值
        outlier_mask = distances > threshold
        new_outlier_indices = set(torch.where(outlier_mask)[0].tolist())
        new_outlier_indices = new_outlier_indices - self.outlier_indices
        
        # 检查是否达到最大剔除比例
        total_observations = len(self.joint_positions)
        current_outlier_ratio = len(self.outlier_indices) / total_observations
        max_additional_outliers = int(total_observations * self.max_outlier_ratio) - len(self.outlier_indices)
        
        if max_additional_outliers <= 0:
            if verbose:
                print(f"已达到最大剔除比例 {self.max_outlier_ratio*100}%，跳过离群值检测")
            return 0
        
        # 限制本轮最多剔除数量
        if len(new_outlier_indices) > max_additional_outliers:
            outlier_distances = [(i, distances[i].item()) for i in new_outlier_indices]
            outlier_distances.sort(key=lambda x: x[1], reverse=True)
            new_outlier_indices = set([x[0] for x in outlier_distances[:max_additional_outliers]])
        
        if len(new_outlier_indices) > self.max_outliers_per_round:
            outlier_distances = [(i, distances[i].item()) for i in new_outlier_indices]
            outlier_distances.sort(key=lambda x: x[1], reverse=True)
            new_outlier_indices = set([x[0] for x in outlier_distances[:self.max_outliers_per_round]])
        
        # 更新离群值索引
        self.outlier_indices.update(new_outlier_indices)
        
        if verbose and len(new_outlier_indices) > 0:
            print(f"检测到 {len(new_outlier_indices)} 个新离群值，总共剔除 {len(self.outlier_indices)} 个观测")
            print(f"当前剔除比例: {len(self.outlier_indices)/total_observations*100:.1f}%")
            print(f"距离统计: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, 阈值={threshold:.4f}")
        
        return len(new_outlier_indices)
    
    def _compute_sphere_centers_in_base(self) -> List[torch.Tensor]:
        """计算所有观测中球心在基坐标系下的位置"""
        sphere_centers_base = []
        
        for i in range(len(self.joint_positions)):
            q = self.joint_positions[i]
            sphere_center_sensor = self.sphere_centers_sensor[i]
            
            T_sensor_to_base = self.robot.fkine(q)
            sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
                torch.tensor(sphere_center_sensor, dtype=torch.float64) if not isinstance(sphere_center_sensor, torch.Tensor) else sphere_center_sensor, 
                torch.tensor([1.0], dtype=torch.float64)
            ])
            
            sphere_center_base_calculated = sphere_center_base_calculated[:3]
            sphere_centers_base.append(sphere_center_base_calculated)
        
        return sphere_centers_base
    
    def _compute_variance_error(self) -> torch.Tensor:
        """
        计算观测点位置的方差，自动选择最优计算方法
        
        根据数据量大小自动选择计算策略：
        - 大数据量(>200): 并行计算 -> 向量化计算 -> 批量计算
        - 中等数据量(5-200): 向量化计算 -> 批量计算
        - 小数据量(<=5): 批量计算
        
        Returns:
            位置方差的总和
        """
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        # 获取有效观测数据（排除离群值）
        valid_indices = [i for i in range(len(self.joint_positions)) if i not in self.outlier_indices]
        joint_positions = [self.joint_positions[i] for i in valid_indices]
        sphere_centers = [self.sphere_centers_sensor[i] for i in valid_indices]
        n_observations = len(joint_positions)
        
        # 根据数据量选择计算策略
        if n_observations > self.parallel_threshold:
            try:
                return self._compute_variance_error_parallel_optimized(joint_positions, sphere_centers)
            except Exception as e:
                print(f"Parallel computation failed: {e}, falling back to vectorized")
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
        elif n_observations > self.vectorized_threshold:
            try:
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
            except Exception as e:
                print(f"Vectorized computation failed: {e}, falling back to batch")
                return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)
        else:
            return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)
    
    def _compute_variance_error_batch_optimized(self, joint_positions, sphere_centers) -> torch.Tensor:
        """优化的批量计算方差误差"""
        n_observations = len(joint_positions)
        
        joint_positions_batch = torch.stack(joint_positions)
        sphere_centers_batch = torch.stack(sphere_centers)
        
        calculated_positions = []
        
        for i in range(n_observations):
            T_sensor_to_base = self.robot.fkine(joint_positions_batch[i])
            sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
                sphere_centers_batch[i], 
                torch.tensor([1.0], dtype=torch.float64)
            ])
            calculated_positions.append(sphere_center_base_calculated[:3])
        
        positions_tensor = torch.stack(calculated_positions)
        variance = torch.var(positions_tensor, dim=0).sum()
        
        return variance
    
    def _compute_variance_error_vectorized_optimized(self, joint_positions, sphere_centers) -> torch.Tensor:
        """优化的向量化计算方差误差"""
        try:
            joint_positions_batch = torch.stack(joint_positions)
            sphere_centers_batch = torch.stack(sphere_centers)

            if hasattr(self.robot, 'fkine_batch'):
                T_batch = self.robot.fkine_batch(joint_positions_batch)
                ones = torch.ones(len(sphere_centers_batch), 1, dtype=torch.float64)
                homogeneous_centers = torch.cat([sphere_centers_batch, ones], dim=1)

                transformed_centers = torch.bmm(T_batch, homogeneous_centers.unsqueeze(-1)).squeeze(-1)
                calculated_positions = transformed_centers[:, :3]
            else:
                return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)

            variance = torch.var(calculated_positions, dim=0).sum()
            return variance

        except Exception as e:
            return self._compute_variance_error_batch_optimized(joint_positions, sphere_centers)
    
    def _compute_variance_error_parallel_optimized(self, joint_positions, sphere_centers) -> torch.Tensor:
        """
        优化的并行计算方差误差
        
        使用ThreadPoolExecutor进行并行计算，适用于大数据量场景
        工作线程数根据数据量自动调整，避免过度并行化
        
        Args:
            joint_positions: 关节位置列表
            sphere_centers: 球心位置列表
            
        Returns:
            位置方差的总和
        """
        n_observations = len(joint_positions)
        
        if n_observations > self.parallel_threshold:
            try:
                # 动态调整工作线程数
                max_workers = min(self.max_parallel_workers, 
                                max(2, n_observations // 50))
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for joint_pos, sphere_center in zip(joint_positions, sphere_centers):
                        future = executor.submit(self._compute_single_position, (joint_pos, sphere_center))
                        futures.append(future)
                    
                    calculated_positions = [future.result() for future in futures]
                
                positions_tensor = torch.stack(calculated_positions)
                variance = torch.var(positions_tensor, dim=0).sum()
                return variance
            except Exception as e:
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
        else:
            return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
    
    def _get_calibration_parameters(self) -> List[torch.Tensor]:
        """获取需要校准的参数"""
        params = []
        
        if hasattr(self.robot, 'links'):
            for i, link in enumerate(self.robot.links):
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        if i in self.excluded_params and param_name in self.excluded_params[i]:
                            continue
                        
                        param = getattr(link, param_name)
                        param.requires_grad = True
                        params.append(param)
        
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                if hasattr(self.robot, param_name):
                    param = getattr(self.robot, param_name)
                    param.requires_grad = True
                    params.append(param)
        
        return params
    
    def calibrate(self, verbose: bool = None) -> Dict:
        """
        执行高精度参数校准
        
        使用AdamW优化器和ReduceLROnPlateau学习率调度器进行优化
        支持梯度裁剪、早停、离群值检测等高级功能
        
        Args:
            verbose: 是否输出详细信息，None时使用配置中的设置
            
        Returns:
            包含校准结果的字典
        """
        if len(self.joint_positions) == 0:
            raise ValueError("No calibration data, please set observations first")
        
        # 使用配置中的verbose设置
        if verbose is None:
            verbose = self.verbose
        
        # ========== 记录初始状态 ==========
        initial_params = self._save_current_params()
        initial_error = self._compute_variance_error().item()
        self.sphere_centers_base_before = self._compute_sphere_centers_in_base()
        
        # ========== 获取待优化参数 ==========
        params = self._get_calibration_parameters()
        if len(params) == 0:
            raise ValueError("No parameters found for calibration")
        
        # ========== 初始化优化器和调度器 ==========
        self.optimization_history = []
        optimizer = optim.AdamW(params, 
                              lr=self.learning_rate, 
                              weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.scheduler_factor, 
            patience=self.scheduler_patience
        )
        
        # ========== 输出训练配置信息 ==========
        if verbose:
            self._print_training_config(initial_error)
        
        # ========== 执行初始离群值检测 ==========
        if self.enable_outlier_detection:
            if verbose:
                print("执行初始离群值检测...")
            initial_outliers = self._detect_and_remove_outliers(verbose)
            if initial_outliers > 0 and verbose:
                print(f"初始检测剔除了 {initial_outliers} 个离群值")
                initial_error_after_outlier_removal = self._compute_variance_error().item()
                print(f"剔除离群值后的初始方差误差: {initial_error_after_outlier_removal:.6f}")
        else:
            if verbose:
                print("离群值检测已禁用")
        
        # ========== 主训练循环 ==========
        prev_loss = float('inf')
        iteration = 0
        
        for iteration in range(self.max_iterations):
            # 前向传播
            optimizer.zero_grad()
            variance_error = self._compute_variance_error()
            current_loss = variance_error.item()
            
            # 定期执行离群值检测
            if (self.enable_outlier_detection and 
                iteration > 0 and 
                iteration % self.outlier_detection_interval == 0):
                removed_count = self._detect_and_remove_outliers(verbose)
                if removed_count > 0 and verbose:
                    print(f"Iteration {iteration}: 剔除了 {removed_count} 个离群值")
            
            # 检查是否达到目标精度
            if current_loss < self.target_precision:
                if verbose:
                    print(f"Target precision achieved at iteration {iteration}")
                break
            
            # 检查是否收敛
            if abs(prev_loss - current_loss) < self.convergence_tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # 反向传播和优化
            variance_error.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.gradient_clip_norm)
            optimizer.step()
            scheduler.step(current_loss)
            
            # 记录历史
            self.optimization_history.append(current_loss)
            
            # 输出训练日志
            if verbose and iteration % self.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                valid_obs_count = len(self.joint_positions) - len(self.outlier_indices)
                print(f"Iteration {iteration}: Variance Error = {current_loss:.16f}, "
                      f"LR = {current_lr:.12f}, Valid Obs = {valid_obs_count}")
            
            prev_loss = current_loss
        
        # ========== 保存最终结果 ==========
        self.sphere_centers_base_after = self._compute_sphere_centers_in_base()
        final_params = self._save_current_params()
        final_variance_error = self._compute_variance_error().item()
        precision_achieved = final_variance_error < self.target_precision
        
        # ========== 构建结果字典 ==========
        self.calibration_results = {
            'initial_params': initial_params,
            'final_params': final_params,
            'initial_variance_error': initial_error,
            'final_variance_error': final_variance_error,
            'target_precision': self.target_precision,
            'precision_achieved': precision_achieved,
            'iterations': len(self.optimization_history),
            'converged': iteration < self.max_iterations - 1,
            'calibration_time': datetime.now().isoformat(),
            'optimization_history': self.optimization_history,
            'sphere_centers_base_before': [center.tolist() for center in self.sphere_centers_base_before],
            'sphere_centers_base_after': [center.tolist() for center in self.sphere_centers_base_after],
            'outlier_info': {
                'total_outliers_removed': len(self.outlier_indices),
                'outlier_ratio': len(self.outlier_indices) / len(self.joint_positions) if len(self.joint_positions) > 0 else 0,
                'outlier_indices': list(self.outlier_indices),
                'final_valid_observations': len(self.joint_positions) - len(self.outlier_indices)
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'max_iterations': self.max_iterations,
                'weight_decay': self.weight_decay,
                'gradient_clip_norm': self.gradient_clip_norm,
                'outlier_detection_enabled': self.enable_outlier_detection
            }
        }
        
        # ========== 输出最终结果 ==========
        if verbose:
            self._print_final_results(initial_error, final_variance_error, precision_achieved)
        
        return self.calibration_results
    
    def _save_current_params(self) -> Dict:
        """保存当前参数状态"""
        params = {}
        
        # 保存连杆参数
        if hasattr(self.robot, 'links'):
            for i, link in enumerate(self.robot.links):
                params[f'link_{i}'] = {}
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        # 保存手眼参数
        params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                if hasattr(self.robot, param_name):
                    params['hand_eye'][param_name] = getattr(self.robot, param_name).item()
        
        return params
    
    def _print_training_config(self, initial_error: float):
        """打印训练配置信息"""
        print(f"Initial Variance Error: {initial_error:.6f}")
        print(f"Target precision: {self.target_precision:.10f}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Gradient clip norm: {self.gradient_clip_norm}")
        
        n_obs = len(self.joint_positions)
        if n_obs > self.parallel_threshold:
            print(f"Data size: {n_obs}, using parallel computation for acceleration")
        elif n_obs > self.vectorized_threshold:
            print(f"Data size: {n_obs}, using vectorized computation for acceleration")
        else:
            print(f"Data size: {n_obs}, using batch computation")
    
    def _print_final_results(self, initial_error: float, final_error: float, precision_achieved: bool):
        """打印最终结果"""
        print(f"\nCalibration completed!")
        print(f"Initial Variance Error: {initial_error:.6f}")
        print(f"Final Variance Error: {final_error:.16f}")
        print(f"Target precision: {self.target_precision:.16f}")
        print(f"Precision achieved: {'Yes' if precision_achieved else 'No'}")
        print(f"Outliers removed: {len(self.outlier_indices)} ({len(self.outlier_indices)/len(self.joint_positions)*100:.1f}%)")
        print(f"Final valid observations: {len(self.joint_positions) - len(self.outlier_indices)}")
        
        if self.enable_outlier_detection:
            if len(self.outlier_indices) > 0:
                sorted_outlier_indices = sorted(list(self.outlier_indices))
                print(f"Removed outlier indices: {sorted_outlier_indices}")
                
                if len(sorted_outlier_indices) > 20:
                    print(f"First 10 outlier indices: {sorted_outlier_indices[:10]}")
                    print(f"Last 10 outlier indices: {sorted_outlier_indices[-10:]}")
            else:
                print("No outliers were removed during calibration.")
        else:
            print("Outlier detection was disabled for this calibration.")
        
        if initial_error > 0:
            improvement = ((initial_error - final_error) / initial_error * 100)
            print(f"Error Improvement: {improvement:.10f}%")
    
    def save_results(self, filename: str):
        """保存校准结果到result文件夹"""
        if self.calibration_results is None:
            raise ValueError("No calibration results available. Please run calibrate() first.")
        
        import os
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        filepath = os.path.join(result_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.calibration_results, f, indent=2)
        
        print(f"Calibration results saved to {filepath}")
    
    def _compute_single_position(self, joint_sphere_pair):
        """计算单个观测的球心位置(用于并行计算)"""
        joint_pos, sphere_center_sensor = joint_sphere_pair
        
        T_sensor_to_base = self.robot.fkine(joint_pos)
        sphere_center_base_calculated = T_sensor_to_base @ torch.cat([
            sphere_center_sensor, 
            torch.tensor([1.0], dtype=torch.float64)
        ])
        
        return sphere_center_base_calculated[:3]


