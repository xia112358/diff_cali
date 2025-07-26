import torch

def fit_sphere_center_TLS_A(points, max_iter=100, tol=1e-6):
    """
    TLS_A球心拟合算法实现
    参数:
        points : (n,3)张量,点云数据
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回:
        center : (3,)张量, 球心坐标 [x0, y0, z0]
        radius: 球体半径
    """
    # 步骤1: LLS_A初始估计
    n_points = points.size(0)
    A = torch.zeros((n_points, 4), dtype=torch.float64)
    B = torch.zeros(n_points, dtype=torch.float64)

    for i, (x, y, z) in enumerate(points):
        A[i] = torch.tensor([2*x, 2*y, 2*z, -1], dtype=torch.float64)
        B[i] = x**2 + y**2 + z**2

    # 最小二乘求解
    solution = torch.linalg.lstsq(A, B.unsqueeze(1)).solution
    x0, y0, z0, rho = solution[:4, 0]
    r0 = torch.sqrt(x0**2 + y0**2 + z0**2 - rho)
    center = torch.tensor([x0, y0, z0], dtype=torch.float64)

    # 步骤2: Gauss-Newton迭代
    for _ in range(max_iter):
        # 计算残差和Jacobian
        residuals = []
        J = []

        for point in points:
            diff = center - point
            dist = torch.norm(diff)
            residuals.append(dist - r0)

            if dist > 1e-12:  # 避免除零错误
                J_row = [
                    diff[0]/dist, 
                    diff[1]/dist,
                    diff[2]/dist,
                    -1
                ]
            else:
                J_row = [0, 0, 0, -1]
            J.append(J_row)

        residuals = torch.tensor(residuals, dtype=torch.float64)
        J = torch.tensor(J, dtype=torch.float64)

        # 参数更新
        delta = torch.linalg.lstsq(J, residuals.unsqueeze(1)).solution
        delta = delta[:4, 0]
        center -= delta[:3]
        r0 -= delta[3]

        # 收敛检查
        if torch.norm(delta) < tol:
            break

    return center

# 示例用法
if __name__ == "__main__":
    # 生成测试数据(球面点云)
    torch.manual_seed(43)
    true_center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    true_radius = 5.0
    angles = torch.rand(100) * 2 * torch.pi
    elevations = torch.rand(100) * torch.pi

    points = torch.stack([
        true_center[0] + true_radius * torch.sin(elevations) * torch.cos(angles),
        true_center[1] + true_radius * torch.sin(elevations) * torch.sin(angles),
        true_center[2] + true_radius * torch.cos(elevations)
    ], dim=1)

    # 添加噪声
    noise = torch.normal(mean=0.0, std=0.1, size=points.shape, dtype=torch.float64)
    points += noise

    # 拟合球心
    center = fit_sphere_center_TLS_A(points)
    print(f"真实球心: {true_center}, 拟合球心: {center}")
    print(f"球心误差: {torch.norm(true_center - center):.6f} mm")