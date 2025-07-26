import torch

def hand_eye_calibration(robot_poses, measured_points):
    """
    手眼标定主函数
    :param robot_poses: 机器人位姿列表 [R_e_b, t_e_b] 
        R_e_b: 3x3旋转矩阵张量 (基坐标系→末端)
        t_e_b: 3x1平移向量张量 (基坐标系→末端)
    :param measured_points: 测量点坐标列表 [x_s, y_s, z_s] (线结构光坐标系) torch.Tensor
    :return: T_s_e: 4x4手眼变换矩阵张量
    """
    # =================================================================
    # 步骤1：标定旋转矩阵 R_s_e
    # =================================================================
    R_se = calibrate_rotation_matrix(robot_poses, measured_points)
    
    # =================================================================
    # 步骤2：标定平移向量 t_s_e
    # =================================================================
    t_se = calibrate_translation_vector(robot_poses, measured_points, R_se)
    
    # 构建完整变换矩阵
    T_se = torch.eye(4, dtype=torch.float64)
    T_se[:3, :3] = R_se
    T_se[:3, 3] = t_se.flatten()
    
    return T_se

def calibrate_rotation_matrix(poses, points):
    """
    旋转矩阵标定 (公式8)
    :param poses: 机器人位姿列表 (纯平移运动)
    :param points: 对应测量点
    """
    n = len(poses)
    A_list, B_list = [], []
    
    # 构建矩阵方程 R_s_e * A = B
    for i in range(n):
        for j in range(i+1, n):
            # 获取位姿数据
            R_e_b = poses[i][0]  # 旋转矩阵相同（平移运动）
            t_i = poses[i][1]
            t_j = poses[j][1]
            
            # 计算向量差 (公式8右侧)
            B_vec = R_e_b.T @ (t_j - t_i)
            
            # 计算测量点差 (公式8左侧)
            A_vec = points[i] - points[j]
            
            # 确保向量是1D的
            A_list.append(A_vec.flatten())
            B_list.append(B_vec.flatten())
    
    # 转换为矩阵
    A_mat = torch.stack(A_list)  # MxN 
    B_mat = torch.stack(B_list)  # MxN
    
    # 转置得到 NxM 矩阵
    A_mat = A_mat.T  # 3xM
    B_mat = B_mat.T  # 3xM
    
    # SVD分解求解 (公式8): 求解 R_se * A_mat = B_mat
    # 即 R_se = B_mat * A_mat^+ (伪逆)
    U, s, Vt = torch.linalg.svd(A_mat @ B_mat.T)
    R_se = U @ Vt
    
    # 保证旋转矩阵行列式=1
    if torch.det(R_se) < 0:
        U[:, -1] *= -1
        R_se = U @ Vt
        
    return R_se

def calibrate_translation_vector(poses, points, R_se):
    """
    平移向量标定 (公式9)
    :param poses: 机器人位姿列表 (含旋转运动)
    :param points: 对应测量点
    :param R_se: 已标定的旋转矩阵
    """
    n = len(poses)
    A_list, b_list = [], []
    
    # 构建超定方程组 A * t_s_e = b
    for i in range(n):
        for j in range(i+1, n):
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
            
            A_list.append(A_block)
            b_list.append(b_vec.flatten())  # 确保是1D向量
    
    # 转换为矩阵
    A_full = torch.vstack(A_list)  # 3Mx3
    b_full = torch.stack(b_list)   # Mx3, 然后需要reshape
    b_full = b_full.flatten().unsqueeze(1)  # (3M)x1
    
    # 最小二乘法求解
    t_se = torch.linalg.lstsq(A_full, b_full).solution
    return t_se

# ================================
# 使用示例
# ================================
if __name__ == "__main__":
    # 模拟数据生成 (实际应用替换为真实数据)
    num_trans = 10  # 平移运动数据组数
    num_rot = 6     # 旋转运动数据组数
    
    # 1. 生成机器人位姿数据 [R_e_b, t_e_b]
    trans_poses = [
        [torch.eye(3, dtype=torch.float64), torch.rand(3, 1, dtype=torch.float64)] for _ in range(num_trans)
    ]
    
    rot_poses = [
        [torch.rand(3, 3, dtype=torch.float64), torch.rand(3, 1, dtype=torch.float64)] for _ in range(num_rot)
    ]
    
    # 2. 生成测量点数据 [x_s, y_s, z_s]
    trans_points = [torch.rand(3, 1, dtype=torch.float64) for _ in range(num_trans)]
    rot_points = [torch.rand(3, 1, dtype=torch.float64) for _ in range(num_rot)]
    
    # 3. 执行标定
    # 步骤1：用平移数据标定旋转矩阵
    R_se = calibrate_rotation_matrix(trans_poses, trans_points)
    
    # 步骤2：用旋转数据标定平移向量
    t_se = calibrate_translation_vector(rot_poses, rot_points, R_se)
    
    # 4. 构建完整变换矩阵
    T_se = torch.eye(4, dtype=torch.float64)
    T_se[:3, :3] = R_se
    T_se[:3, 3] = t_se.flatten()
    
    print("标定结果 T_s_e:")
    print(torch.round(T_se, decimals=4))