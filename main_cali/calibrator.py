import torch
import torch.optim as optim
from robot_model.unified_robot import BaseRobot
from typing import List, Dict
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class RobotCalibrator:
    """基于位置方差的机器人运动学参数校准类，支持自动微分优化和离群值检测。"""
    
    def __init__(self, robot: BaseRobot, calibration_params: List[str] = None, 
                 excluded_params: Dict[int, List[str]] = None, target_precision: float = 1e-6,
                 enable_outlier_detection: bool = False):
        """初始化校准器"""
        self.robot = robot
        
        if calibration_params is None:
            self.calibration_params = ['a', 'alpha', 'd', 'beta', 'offset', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        else:
            self.calibration_params = calibration_params
        
        if excluded_params is None:
            self.excluded_params = {}
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
        
        self.enable_outlier_detection = enable_outlier_detection
        self.outlier_detection_interval = 50
        self.max_outlier_ratio = 0.3
        self.outlier_threshold = 1.5
        self.max_outliers_per_round = 10
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
        """检测并剔除离群值观测数据"""
        if not self.enable_outlier_detection:
            return 0
        if len(self.joint_positions) == 0:
            return 0
        
        sphere_centers_base = self._compute_sphere_centers_in_base()
        positions_tensor = torch.stack(sphere_centers_base)
        center_position = torch.mean(positions_tensor, dim=0)
        distances = torch.norm(positions_tensor - center_position, dim=1)
        
        q1 = torch.quantile(distances, 0.25)
        q3 = torch.quantile(distances, 0.75)
        iqr = q3 - q1
        threshold = q3 + self.outlier_threshold * iqr
        
        outlier_mask = distances > threshold
        new_outlier_indices = set(torch.where(outlier_mask)[0].tolist())
        new_outlier_indices = new_outlier_indices - self.outlier_indices
        
        total_observations = len(self.joint_positions)
        current_outlier_ratio = len(self.outlier_indices) / total_observations
        max_additional_outliers = int(total_observations * self.max_outlier_ratio) - len(self.outlier_indices)
        
        if max_additional_outliers <= 0:
            if verbose:
                print(f"已达到最大剔除比例 {self.max_outlier_ratio*100}%，跳过离群值检测")
            return 0
        
        if len(new_outlier_indices) > max_additional_outliers:
            outlier_distances = [(i, distances[i].item()) for i in new_outlier_indices]
            outlier_distances.sort(key=lambda x: x[1], reverse=True)
            new_outlier_indices = set([x[0] for x in outlier_distances[:max_additional_outliers]])
        
        if len(new_outlier_indices) > self.max_outliers_per_round:
            outlier_distances = [(i, distances[i].item()) for i in new_outlier_indices]
            outlier_distances.sort(key=lambda x: x[1], reverse=True)
            new_outlier_indices = set([x[0] for x in outlier_distances[:self.max_outliers_per_round]])
        
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
        """计算观测点位置的方差，自动选择最优计算方法"""
        if len(self.joint_positions) == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        
        valid_indices = [i for i in range(len(self.joint_positions)) if i not in self.outlier_indices]
        joint_positions = [self.joint_positions[i] for i in valid_indices]
        sphere_centers = [self.sphere_centers_sensor[i] for i in valid_indices]
        n_observations = len(joint_positions)
        
        if n_observations > 100:
            try:
                return self._compute_variance_error_parallel_optimized(joint_positions, sphere_centers)
            except Exception as e:
                print(f"Parallel computation failed: {e}, falling back to vectorized")
                return self._compute_variance_error_vectorized_optimized(joint_positions, sphere_centers)
        elif n_observations > 5:
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
        """优化的并行计算方差误差"""
        n_observations = len(joint_positions)
        
        if n_observations > 200:
            try:
                max_workers = min(4, max(2, n_observations // 50))
                
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
    
    def calibrate(self, learning_rate: float = None, max_iterations: int = None, 
                  tolerance: float = None, verbose: bool = True) -> Dict:
        """执行高精度参数校准"""
        if len(self.joint_positions) == 0:
            raise ValueError("No calibration data, please set observations first")
        
        lr = learning_rate if learning_rate is not None else self.learning_rate
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        tol = tolerance if tolerance is not None else self.tolerance
        
        initial_params = {}
        if hasattr(self.robot, 'links'):
            for i, link in enumerate(self.robot.links):
                initial_params[f'link_{i}'] = {}
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        initial_params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        initial_params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                if hasattr(self.robot, param_name):
                    initial_params['hand_eye'][param_name] = getattr(self.robot, param_name).item()
        
        initial_error = self._compute_variance_error().item()
        self.sphere_centers_base_before = self._compute_sphere_centers_in_base()
        
        params = self._get_calibration_parameters()
        
        if len(params) == 0:
            raise ValueError("No parameters found for calibration")
        
        self.optimization_history = []
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=50
        )
        
        if verbose:
            print(f"Initial Variance Error: {initial_error:.6f}")
            print(f"Target precision: {self.target_precision:.10f}")
            print(f"Using full-batch training")
            
            n_obs = len(self.joint_positions)
            if n_obs > 100:
                print(f"Data size: {n_obs}, using parallel computation for acceleration")
            elif n_obs > 20:
                print(f"Data size: {n_obs}, using vectorized computation for acceleration")
            else:
                print(f"Data size: {n_obs}, using batch computation")
        
        if self.enable_outlier_detection:
            print("执行初始离群值检测...")
            initial_outliers = self._detect_and_remove_outliers(verbose)
            if initial_outliers > 0 and verbose:
                print(f"初始检测剔除了 {initial_outliers} 个离群值")
                initial_error_after_outlier_removal = self._compute_variance_error().item()
                print(f"剔除离群值后的初始方差误差: {initial_error_after_outlier_removal:.6f}")
        else:
            if verbose:
                print("离群值检测已禁用")
        
        prev_loss = float('inf')
        for iteration in range(max_iter):
            optimizer.zero_grad()
            variance_error = self._compute_variance_error()
            current_loss = variance_error.item()
            full_loss = current_loss
            
            if self.enable_outlier_detection and iteration > 0 and iteration % self.outlier_detection_interval == 0:
                removed_count = self._detect_and_remove_outliers(verbose)
                if removed_count > 0 and verbose:
                    print(f"Iteration {iteration}: 剔除了 {removed_count} 个离群值")
            
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
                valid_obs_count = len(self.joint_positions) - len(self.outlier_indices)
                print(f"Iteration {iteration}: Variance Error = {current_loss:.16f}, LR = {current_lr:.12f}, Valid Obs = {valid_obs_count}")
            prev_loss = current_loss
        
        self.sphere_centers_base_after = self._compute_sphere_centers_in_base()
        
        final_params = {}
        if hasattr(self.robot, 'links'):
            for i, link in enumerate(self.robot.links):
                final_params[f'link_{i}'] = {}
                for param_name in self.calibration_params:
                    if hasattr(link, param_name):
                        final_params[f'link_{i}'][param_name] = getattr(link, param_name).item()
        
        final_params['hand_eye'] = {}
        hand_eye_param_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for param_name in hand_eye_param_names:
            if param_name in self.calibration_params:
                if hasattr(self.robot, param_name):
                    final_params['hand_eye'][param_name] = getattr(self.robot, param_name).item()
        
        final_variance_error = self._compute_variance_error().item()
        precision_achieved = final_variance_error < self.target_precision
        
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
            'optimization_history': self.optimization_history,
            'sphere_centers_base_before': [center.tolist() for center in self.sphere_centers_base_before],
            'sphere_centers_base_after': [center.tolist() for center in self.sphere_centers_base_after],
            'outlier_info': {
                'total_outliers_removed': len(self.outlier_indices),
                'outlier_ratio': len(self.outlier_indices) / len(self.joint_positions) if len(self.joint_positions) > 0 else 0,
                'outlier_indices': list(self.outlier_indices),
                'final_valid_observations': len(self.joint_positions) - len(self.outlier_indices)
            }
        }
        
        if verbose:
            print(f"\nCalibration completed!")
            print(f"Initial Variance Error: {initial_error:.6f}")
            print(f"Final Variance Error: {final_variance_error:.16f}")
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
                improvement = ((initial_error - final_variance_error) / initial_error * 100)
                print(f"Error Improvement: {improvement:.10f}%")
    
        return self.calibration_results
    
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


