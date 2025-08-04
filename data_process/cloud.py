import torch

def fit_sphere_center_TLS_A(points, max_iter=100, tol=1e-6, device='cuda'):
    """
    TLS_A球心拟合算法实现（GPU并行加速版本）
    参数:
        points : (n,3)张量,点云数据
        max_iter: 最大迭代次数
        tol: 收敛容差
        device: 计算设备，默认使用'cuda'
    返回:
        center : (3,)张量, 球心坐标 [x0, y0, z0]
    """
    # 如果GPU不可用且指定了cuda，自动回退到CPU
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("GPU不可用，自动切换到CPU计算")
    
    # 将数据移到指定设备并转换为双精度
    points = points.to(device).to(torch.float64)
    n_points = points.size(0)
    
    # 步骤1: LLS_A初始估计（向量化实现）
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # 构建矩阵A和向量B（并行计算）
    A = torch.stack([2*x, 2*y, 2*z, -torch.ones_like(x)], dim=1)
    B = x**2 + y**2 + z**2
    
    # 最小二乘求解
    solution = torch.linalg.lstsq(A, B.unsqueeze(1)).solution
    x0, y0, z0, rho = solution[:4, 0]
    r0 = torch.sqrt(x0**2 + y0**2 + z0**2 - rho)
    center = torch.tensor([x0, y0, z0], dtype=torch.float64, device=device)
    
    # 步骤2: Gauss-Newton迭代（优化的向量化实现）
    for iteration in range(max_iter):
        # 计算所有点到球心的差值（并行计算）
        diff = center.unsqueeze(0) - points  # (n_points, 3)
        
        # 计算距离（并行计算，使用更稳定的计算方式）
        distances = torch.norm(diff, dim=1, keepdim=False)  # (n_points,)
        
        # 计算残差（并行计算）
        residuals = distances - r0  # (n_points,)
        
        # 计算Jacobian矩阵（向量化实现，改进数值稳定性）
        # 使用更大的阈值避免数值不稳定
        safe_mask = distances > 1e-8
        safe_distances = distances.clone()
        safe_distances[~safe_mask] = 1.0  # 临时设置为1避免除零
        
        # Jacobian矩阵的前3列：diff / distance
        J_xyz = diff / safe_distances.unsqueeze(1)  # (n_points, 3)
        
        # 处理距离过小的特殊情况
        J_xyz[~safe_mask] = 0.0
        
        # Jacobian矩阵的第4列：-1
        J_r = -torch.ones((n_points, 1), dtype=torch.float64, device=device)
        
        # 组合完整的Jacobian矩阵
        J = torch.cat([J_xyz, J_r], dim=1)  # (n_points, 4)
        
        # 参数更新（使用改进的数值稳定性）
        try:
            # 使用更稳定的求解方法
            JTJ = J.T @ J
            JTr = J.T @ residuals
            
            # 添加正则化项防止奇异矩阵
            reg_term = 1e-8 * torch.eye(4, dtype=torch.float64, device=device)
            delta = torch.linalg.solve(JTJ + reg_term, JTr)
            
            # 更新参数
            center = center - delta[:3]
            r0 = r0 - delta[3]
            
            # 收敛检查
            delta_norm = torch.norm(delta)
            if delta_norm < tol:
                # 静默收敛，只在调试时输出
                # print(f"TLS_A算法在第{iteration+1}次迭代后收敛，delta_norm={delta_norm:.8f}")
                break
                
        except Exception as e:
            print(f"TLS_A算法在第{iteration+1}次迭代时出现数值问题: {e}")
            break
    
    # 返回到CPU（如果原本在GPU）
    return center.cpu() if center.is_cuda else center

# 示例用法和性能测试
if __name__ == "__main__":
    import time
    import open3d as o3d
    import numpy as np
    import os
    
    print("=== TLS_A球心拟合算法测试 ===")
    
    # 测试真实点云数据
    ply_file = "data/clouds/point_cloud_00001.ply"
    if os.path.exists(ply_file):
        print(f"\n=== 读取真实点云: {ply_file} ===")
        
        # 使用Open3D读取PLY文件
        pcd = o3d.io.read_point_cloud(ply_file)
        real_points = np.asarray(pcd.points)
        
        print(f"点云信息:")
        print(f"  原始点数: {len(real_points)}")
        print(f"  坐标范围: X[{real_points[:, 0].min():.3f}, {real_points[:, 0].max():.3f}]")
        print(f"           Y[{real_points[:, 1].min():.3f}, {real_points[:, 1].max():.3f}]")
        print(f"           Z[{real_points[:, 2].min():.3f}, {real_points[:, 2].max():.3f}]")
        
        # 检查数据有效性
        if len(real_points) < 10:
            print("❌ 有效点云数据不足，无法进行球心拟合")
        else:
            # 转换为torch张量
            points_tensor = torch.tensor(real_points, dtype=torch.float64)
            
            # 测试球心拟合
            print(f"\n开始球心拟合...")
            start_time = time.time()
            center_result = fit_sphere_center_TLS_A(points_tensor)
            fit_time = time.time() - start_time
            
            print(f"拟合结果:")
            print(f"  球心坐标: [{center_result[0]:.3f}, {center_result[1]:.3f}, {center_result[2]:.3f}]")
            print(f"  拟合时间: {fit_time:.4f}s")
            
            # 计算拟合质量
            distances = torch.norm(points_tensor - center_result.unsqueeze(0), dim=1)
            fitted_radius = torch.mean(distances)
            radius_std = torch.std(distances)
            
            print(f"  拟合半径: {fitted_radius:.3f}mm")
            print(f"  半径标准差: {radius_std:.3f}mm (越小越好)")
            print(f"  拟合质量: {'良好' if radius_std < 1.0 else '一般' if radius_std < 5.0 else '较差'}")
        
    else:
        print(f"⚠️ 点云文件不存在: {ply_file}")
        print("使用模拟数据进行测试...")
        
        # 生成测试数据(球面点云)
        torch.manual_seed(43)
        true_center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        true_radius = 5.0
        
        # 测试不同数据规模
        test_sizes = [1000, 10000, 100000]
        
        for n_points in test_sizes:
            print(f"\n--- 测试点云规模: {n_points} 个点 ---")
            
            # 生成球面点云
            angles = torch.rand(n_points) * 2 * torch.pi
            elevations = torch.rand(n_points) * torch.pi
            
            points = torch.stack([
                true_center[0] + true_radius * torch.sin(elevations) * torch.cos(angles),
                true_center[1] + true_radius * torch.sin(elevations) * torch.sin(angles),
                true_center[2] + true_radius * torch.cos(elevations)
            ], dim=1)
            
            # 添加噪声
            noise = torch.normal(mean=0.0, std=0.1, size=points.shape, dtype=torch.float64)
            points += noise
            
            # GPU计算测试（默认）
            print("GPU计算:")
            start_time = time.time()
            center_gpu = fit_sphere_center_TLS_A(points)  # 默认使用GPU
            gpu_time = time.time() - start_time
            gpu_error = torch.norm(true_center - center_gpu)
            print(f"  时间: {gpu_time:.4f}s, 球心误差: {gpu_error:.6f}")
            
            # CPU对比测试
            print("CPU对比:")
            start_time = time.time()
            center_cpu = fit_sphere_center_TLS_A(points, device='cpu')
            cpu_time = time.time() - start_time
            cpu_error = torch.norm(true_center - center_cpu)
            print(f"  时间: {cpu_time:.4f}s, 球心误差: {cpu_error:.6f}")
            
            if torch.cuda.is_available():
                print(f"  GPU加速比: {cpu_time/gpu_time:.2f}x")
        
        print("\n=== 算法精度验证 ===")
        # 精度测试 - 使用模拟数据的最后一组
        points_small = points[:1000]  # 使用较小的数据集进行精度测试
        
        center_result = fit_sphere_center_TLS_A(points_small)  # 默认GPU计算
        
        print(f"真实球心: {true_center}")
        print(f"拟合球心: {center_result}")
        print(f"球心误差: {torch.norm(true_center - center_result):.6f}")
        
        # 计算拟合半径
        distances = torch.norm(points_small - center_result.unsqueeze(0), dim=1)
        fitted_radius = torch.mean(distances)
        
        print(f"真实半径: {true_radius:.6f}")
        print(f"拟合半径: {fitted_radius:.6f} (误差: {abs(true_radius - fitted_radius):.6f})")
    
    print("\n=== 性能优化总结 ===")
    print("✅ GPU并行计算加速（默认使用CUDA）")
    print("✅ 数值稳定性改进（正则化，更好的阈值）")
    print("✅ 高精度TLS_A算法（适用于精密标定）")
    print("✅ 自动GPU回退（GPU不可用时自动用CPU）")
    print("✅ 向量化并行计算")
    
    print(f"\n💡 使用方法:")
    print(f"  - 默认GPU计算: fit_sphere_center_TLS_A(points)")
    print(f"  - 强制CPU计算: fit_sphere_center_TLS_A(points, device='cpu')")
    print(f"  - 手眼标定场景: 优先使用GPU获得最佳性能")