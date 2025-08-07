import numpy as np
import torch
import json
from robot_model.robots import create_robot


class PreCalibrator:
    """手眼预标定器类"""
    
    def __init__(self):
        """初始化预标定器"""
        self.robot = None
        self.observations = None
        
    def set_observations(self, observations, max_observations: int = None):
        """
        设置观测数据 - 适配新的数据格式
        
        参数:
            observations: 已处理的观测数据列表
            max_observations: 最大读取观测数据个数，None表示读取全部
            新格式: 包含 joint_state 和 point_cloud_center
            
        返回:
            bool: 设置是否成功
        """
        if observations is None:
            print("❌ 观测数据不能为空")
            return False
        
        # 限制观测数据数量
        if max_observations is not None and max_observations > 0:
            observations = observations[:max_observations]
            
        if len(observations) < 12:
            print(f"❌ 观测数据不足，需要至少12组，当前只有{len(observations)}组")
            return False
            
        self.observations = observations
        print(f"✓ 成功设置 {len(observations)} 个观测数据")
        return True
    
    def load_observations_from_json(self, json_file_path: str, max_observations: int = None):
        """
        从JSON文件加载观测数据
        
        参数:
            json_file_path: observations.json文件路径
            max_observations: 最大读取观测数据个数，None表示读取全部
            
        返回:
            bool: 加载是否成功
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'observations' not in data:
                print("❌ JSON文件必须包含 'observations' 字段")
                return False
            
            observations = data['observations']
            total_observations = len(observations)
            
            success = self.set_observations(observations, max_observations)
            if success:
                if max_observations is not None and max_observations < total_observations:
                    print(f"✓ 成功从 {json_file_path} 加载 {len(self.observations)} 个观测数据 (共 {total_observations} 个)")
                else:
                    print(f"✓ 成功从 {json_file_path} 加载观测数据")
            return success
            
        except Exception as e:
            print(f"❌ 从JSON文件加载观测数据失败: {str(e)}")
            return False
        
    def _joint_angles_to_poses(self, joint_angles_list, robot=None):
        """
        将关节角度列表转换为末端位姿
        
        参数:
            joint_angles_list: 关节角度列表，每个元素包含6个关节角度(弧度制)
            robot: 机器人模型，如果为None则创建FR16校准机器人
            
        返回:
            poses_list: 位姿列表，每个元素包含[旋转矩阵, 平移向量]
        """
        if robot is None:
            if self.robot is None:
                self.robot = create_robot(robot_name="FR16")  # 创建FR16机器人
            robot = self.robot
        
        poses_list = []
        
        for joint_angles in joint_angles_list:
            # 确保关节角度是torch tensor
            if not isinstance(joint_angles, torch.Tensor):
                joint_angles = torch.tensor(joint_angles, dtype=torch.float64)
            
            # 计算正运动学
            T_end = robot.fkine(joint_angles)
            
            # 提取旋转矩阵和平移向量，转换为numpy
            if isinstance(T_end, torch.Tensor):
                R_e_b = T_end[:3, :3].detach().numpy()
                t_e_b = T_end[:3, 3].detach().numpy().reshape(-1, 1)
            else:
                # 兼容老版本接口
                R_e_b = T_end[:3, :3]
                t_e_b = T_end[:3, 3].reshape(-1, 1)
            
            poses_list.append([R_e_b, t_e_b])
        
        return poses_list

    def _hand_eye_calibration(self, robot_poses, measured_points):
        """
        手眼标定主函数
        :param robot_poses: 机器人位姿列表 [R_e_b, t_e_b] 
            R_e_b: 3x3旋转矩阵数组 (基坐标系→末端)
            t_e_b: 3x1平移向量数组 (基坐标系→末端)
            前3个位姿用于平移运动标定旋转矩阵，后3个位姿用于旋转运动标定平移向量
        :param measured_points: 测量点坐标列表 [x_s, y_s, z_s] (线结构光坐标系) numpy.ndarray
        :return: T_s_e: 4x4手眼变换矩阵数组
        """
        if len(robot_poses) < 12 or len(measured_points) < 12:
            raise ValueError("需要至少12组数据进行手眼标定")
        
        # 分离数据：前4组用于旋转标定，后8组用于平移标定
        trans_poses = robot_poses[:4]    # 前4组位姿 (平移运动数据)
        trans_points = measured_points[:4]  # 前4组测量点
        
        rot_poses = robot_poses[4:12]      # 后8组位姿 (旋转运动数据)  
        rot_points = measured_points[4:12]    # 后8组测量点
        
        # =================================================================
        # 步骤1：用平移数据标定旋转矩阵 R_s_e
        # =================================================================
        R_se = self._calibrate_rotation_matrix(trans_poses, trans_points)
        
        # =================================================================
        # 步骤2：用旋转数据标定平移向量 t_s_e
        # =================================================================
        t_se = self._calibrate_translation_vector(rot_poses, rot_points, R_se)
        
        # 构建完整变换矩阵
        T_se = np.eye(4, dtype=np.float64)
        T_se[:3, :3] = R_se
        T_se[:3, 3] = t_se.flatten()
        
        return T_se

    def _calibrate_rotation_matrix(self, poses, points):
        """
        旋转矩阵标定 (公式8) - 提高精度版本
        :param poses: 机器人位姿列表 (纯平移运动) - 4个位姿，相邻两组构建一个方程
        :param points: 对应测量点 - 4个测量点
        """
        n = len(poses)
        if n != 4:
            raise ValueError(f"需要4个位姿进行旋转标定，当前有{n}个")
            
        A_list, B_list = [], []
        
        # 构建矩阵方程 R_s_e * A = B
        # 4个位姿相邻两组构建一个方程，构建3个方程
        for i in range(n - 1):  # 相邻配对，构建3个方程
            j = i + 1
                
            # 获取位姿数据
            R_e_b = poses[i][0]  # 旋转矩阵相同（平移运动）
            t_i = poses[i][1]
            t_j = poses[j][1]
            
            # 计算向量差 (公式8右侧)
            B_vec = R_e_b.T @ (t_j - t_i)
            
            # 计算测量点差 (公式8左侧)
            A_vec = points[i] - points[j]
            
            # 数据有效性检查
            '''
            if np.linalg.norm(A_vec) < 1e-8 or np.linalg.norm(B_vec) < 1e-8:
                print(f"警告: 配对 {i+1}-{j+1} 向量差过小，可能影响求解精度")
                continue
            '''
            # 确保向量是1D的
            A_list.append(A_vec.flatten())
            B_list.append(B_vec.flatten())
            
            print(f"相邻配对 {i+1}-{j+1}: A_vec norm={np.linalg.norm(A_vec):.6f}, B_vec norm={np.linalg.norm(B_vec):.6f}")
        
        if len(A_list) < 3:
            raise ValueError("有效的数据对不足3个，无法求解旋转矩阵")
            
        # 转换为矩阵
        A_mat = np.stack(A_list)  # 3x3 (3对*3) 
        B_mat = np.stack(B_list)  # 3x3 (3对*3)
        
        print(f"A_mat shape: {A_mat.shape}, B_mat shape: {B_mat.shape}")
        
        # 转置得到 3x3 矩阵
        A_mat = A_mat.T  # 3x3
        B_mat = B_mat.T  # 3x3
        
        # 使用Procrustes分析求解最优旋转矩阵
        # 求解 R_se * A_mat = B_mat，即 R_se = B_mat * A_mat^T
        H = B_mat @ A_mat.T
        
        # SVD分解求解正交Procrustes问题
        U, s, Vt = np.linalg.svd(H)
        
        # 计算条件数检查求解稳定性
        condition_number = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
        print(f"SVD条件数: {condition_number:.2e}")
        
        if condition_number > 1e6:
            print("警告: 条件数过大，求解可能不稳定")
        
        # 构造旋转矩阵
        R_se = U @ Vt
        
        # 保证旋转矩阵行列式=1（右手坐标系）
        if np.linalg.det(R_se) < 0:
            # 翻转最小奇异值对应的列
            U[:, -1] *= -1
            R_se = U @ Vt
            print("检测到镜像，已修正为正确的旋转矩阵")
        
        # 验证旋转矩阵的正交性
        orthogonality_error = np.linalg.norm(R_se @ R_se.T - np.eye(3))
        print(f"旋转矩阵正交性误差: {orthogonality_error:.2e}")
        
        if orthogonality_error > 1e-6:
            print("警告: 旋转矩阵正交性误差较大")
            
        return R_se

    def _calibrate_translation_vector(self, poses, points, R_se):
        """
        平移向量标定 (公式9) - 提高精度版本
        :param poses: 机器人位姿列表 (含旋转运动) - 8个位姿，两两配对
        :param points: 对应测量点 - 8个测量点
        :param R_se: 已标定的旋转矩阵
        """
        n = len(poses)
        if n != 8:
            raise ValueError(f"需要8个位姿进行平移标定，当前有{n}个")
            
        A_list, b_list = [], []
        
        # 构建超定方程组 A * t_s_e = b
        # 8个位姿任意两个相互配对，构建C(8,2)=28个方程
        for i in range(n):  # 任意两个配对，构建C(8,2)=28个方程
            for j in range(i + 1, n):
                
                # 获取位姿数据
                R_i = poses[i][0]
                R_j = poses[j][0]
                t_i = poses[i][1]
                t_j = poses[j][1]
                
                # 计算系数矩阵 (公式9左侧)
                A_block = R_i - R_j
                
                # 计算常数项 (公式9右侧)
                term1 = R_j @ R_se @ points[j]
                term2 = R_i @ R_se @ points[i]
                b_vec = term1 - term2 + t_j - t_i
                
                # 数据有效性检查
                A_condition = np.linalg.cond(A_block)
                if A_condition > 1e18:
                    print(f"警告: 配对 {i+1}-{j+1} 系数矩阵条件数过大 ({A_condition:.2e})，可能影响求解精度")
                    continue
                
                if np.linalg.norm(b_vec) < 1e-8:
                    print(f"警告: 配对 {i+1}-{j+1} 常数项过小，跳过此配对")
                    continue
                
                A_list.append(A_block)
                b_list.append(b_vec.flatten())  # 确保是1D向量
                
                print(f"配对 {i+1}-{j+1}: A条件数={A_condition:.2e}, b_vec范数={np.linalg.norm(b_vec):.6f}")
        
        if len(A_list) < 3:
            raise ValueError("有效的数据对不足3个，无法求解平移向量")
            
        # 转换为矩阵
        A_full = np.vstack(A_list)  # (N*3)x3 (N对*3行每对，N=有效配对数)
        b_full = np.stack(b_list)   # Nx3, 然后需要reshape
        b_full = b_full.flatten().reshape(-1, 1)  # (N*3)x1
        
        print(f"A_full shape: {A_full.shape}, b_full shape: {b_full.shape}")
        
        # 检查系统条件数
        condition_number = np.linalg.cond(A_full)
        print(f"系统条件数: {condition_number:.2e}")
        
        if condition_number > 1e10:
            print("警告: 系统条件数过大，使用正则化最小二乘法")
            # 使用Tikhonov正则化
            alpha = 1e-6  # 正则化参数
            AtA = A_full.T @ A_full
            Atb = A_full.T @ b_full
            t_se = np.linalg.solve(AtA + alpha * np.eye(3), Atb)
        else:
            # 标准最小二乘法求解
            t_se = np.linalg.lstsq(A_full, b_full, rcond=None)[0]
        
        # 计算残差
        residual = np.linalg.norm(A_full @ t_se - b_full)
        print(f"平移标定残差: {residual:.6f}")
        
        return t_se

    def _extract_sphere_centers(self, observations):
        """
        从观测数据中提取球心坐标 - 新格式
        
        参数:
            observations: 观测数据列表
            新格式: 包含 point_cloud_center 字段，直接使用预计算的球心
            
        返回:
            measured_points: 球心坐标列表
        """
        measured_points = []
        for idx, obs in enumerate(observations):
            try:
                # 检查新格式：直接使用预计算的点云中心
                if 'point_cloud_center' in obs:
                    point_cloud_center = obs['point_cloud_center']
                    if point_cloud_center is not None:
                        # 转换为numpy数组并reshape为列向量
                        sphere_center = np.array(point_cloud_center, dtype=np.float64).reshape(3, 1)
                        print(f"观测 {idx+1}: 使用预计算球心 = [{sphere_center[0,0]:.6f}, {sphere_center[1,0]:.6f}, {sphere_center[2,0]:.6f}]")
                        
                        # 验证球心坐标的合理性
                        if np.any(np.isnan(sphere_center)) or np.any(np.isinf(sphere_center)):
                            print(f"警告: 观测 {idx+1} 球心坐标无效，跳过")
                            continue
                        measured_points.append(sphere_center)
                    else:
                        print(f"警告: 观测 {idx+1} 点云中心为空，跳过")
                        continue
                else:
                    print(f"❌ 观测 {idx+1} 没有 'point_cloud_center' 字段，请使用新格式的观测数据")
                    continue
                
            except Exception as e:
                print(f"警告: 观测 {idx+1} 球心处理失败: {str(e)}")
                continue
                
        print(f"成功提取 {len(measured_points)} 个有效球心坐标")
        return measured_points

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """将旋转矩阵转换为欧拉角(XYZ顺序)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def pre_calibrator(self):
        """
        手眼预标定方法 - 提高精度版本
        :return: 标定结果参数字典
        """
        print("=== 手眼预标定 (高精度版本) ===")

        # 检查是否已设置观测数据
        if self.observations is None:
            print("❌ 尚未设置观测数据，请先调用 set_observations() 方法")
            return None
        
        try:
            # 提取关节角度和测量点
            joint_angles_list = [obs['joint_state'] for obs in self.observations]
            measured_points = self._extract_sphere_centers(self.observations)

            if len(joint_angles_list) < 12 or len(measured_points) < 12:
                print(f"❌ 有效观测数据不足，需要至少12组，当前有{len(measured_points)}组有效数据")
                return None

            # 计算末端位姿 - 使用高精度
            print("计算机器人末端位姿...")
            robot_poses = self._joint_angles_to_poses(joint_angles_list)

            # 分离平移和旋转数据
            trans_poses, trans_points = robot_poses[:4], measured_points[:4]
            rot_poses, rot_points = robot_poses[4:12], measured_points[4:12]

            print(f"平移标定数据: {len(trans_poses)} 组位姿, {len(trans_points)} 个测量点")
            print(f"旋转标定数据: {len(rot_poses)} 组位姿, {len(rot_points)} 个测量点")

            # 标定旋转矩阵
            print("\n=== 步骤1: 标定旋转矩阵 ===")
            R_se = self._calibrate_rotation_matrix(trans_poses, trans_points)
            
            # 标定平移向量
            print("\n=== 步骤2: 标定平移向量 ===")
            t_se = self._calibrate_translation_vector(rot_poses, rot_points, R_se)

            # 构建变换矩阵
            T_se = np.eye(4, dtype=np.float64)
            T_se[:3, :3] = R_se
            T_se[:3, 3] = t_se.flatten()

            # 提取标定结果参数
            calibrated_translation = T_se[:3, 3]
            calibrated_rotation_matrix = T_se[:3, :3]
            calibrated_euler = self._rotation_matrix_to_euler(calibrated_rotation_matrix)

            # 输出标定结果
            print("\n=== 标定结果 ===")
            print(f"平移向量: [{calibrated_translation[0]:.6f}, {calibrated_translation[1]:.6f}, {calibrated_translation[2]:.6f}] (mm)")
            print(f"欧拉角: [{np.degrees(calibrated_euler[0]):.3f}, {np.degrees(calibrated_euler[1]):.3f}, {np.degrees(calibrated_euler[2]):.3f}] (度)")
            
            # 计算标定质量指标
            print("\n=== 标定质量评估 ===")
            rotation_det = np.linalg.det(calibrated_rotation_matrix)
            rotation_orthogonality = np.linalg.norm(calibrated_rotation_matrix @ calibrated_rotation_matrix.T - np.eye(3))
            print(f"旋转矩阵行列式: {rotation_det:.8f} (理想值: 1.0)")
            print(f"旋转矩阵正交性误差: {rotation_orthogonality:.8f} (理想值: 0.0)")

            # 返回标定结果
            return {
                'transform_matrix': T_se,
                'translation': calibrated_translation,
                'rotation_matrix': calibrated_rotation_matrix,
                'euler_angles': calibrated_euler,
                'quality_metrics': {
                    'rotation_determinant': rotation_det,
                    'orthogonality_error': rotation_orthogonality
                }
            }

        except Exception as e:
            print(f"❌ 手眼标定过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None