from typing import Dict
import numpy as np      
import matplotlib.pyplot as plt






def plot_calibration_results(calibration_results: Dict, true_sphere_center=None):
    """
    绘制校准前后的点云中心在基坐标系下的三维图

    :param calibration_results: 校准结果字典,包含校准前后的点云中心和误差信息
    :param true_sphere_center: 真实球心位置 [x, y, z]，可选参数，如果不提供则使用校准后数据的质心
    """
    # 从校准结果中提取数据
    sphere_centers_base_before = calibration_results.get('sphere_centers_base_before', [])
    sphere_centers_base_after = calibration_results.get('sphere_centers_base_after', [])
    target_precision = calibration_results.get('target_precision', 1.0)

    if not sphere_centers_base_before or not sphere_centers_base_after:
        print("Error: Missing calibration data. Please provide valid calibration_results with sphere centers.")
        return
    
    # 确定参考中心点
    if true_sphere_center is not None:
        reference_center = np.array(true_sphere_center)
        reference_label = 'True Center'
    else:
        # 使用校准后数据的质心作为参考
        after_points = np.array([center for center in sphere_centers_base_after])
        reference_center = np.mean(after_points, axis=0)
        reference_label = 'Centroid (After)'

    # 创建3D图
    fig = plt.figure(figsize=(15, 5))

    # 校准前的图
    ax1 = fig.add_subplot(131, projection='3d')

    # 转换为numpy数组用于绘图
    before_points = np.array([center for center in sphere_centers_base_before])

    # 绘制校准前的点
    ax1.scatter(before_points[:, 0], before_points[:, 1], before_points[:, 2],
                c='red', s=50, alpha=0.7, label='Calculated Centers (Before)')

    # 绘制参考中心
    ax1.scatter(reference_center[0], reference_center[1], reference_center[2],
                c='blue', s=100, marker='*', label=reference_label)

    # 绘制连接线
    for point in before_points:
        ax1.plot([point[0], reference_center[0]], [point[1], reference_center[1]],
                 [point[2], reference_center[2]], 'r--', alpha=0.3)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Before Calibration')
    ax1.legend()

    # 校准后的图
    ax2 = fig.add_subplot(132, projection='3d')

    # 转换为numpy数组用于绘图
    after_points = np.array([center for center in sphere_centers_base_after])

    # 绘制校准后的点
    ax2.scatter(after_points[:, 0], after_points[:, 1], after_points[:, 2],
                c='green', s=50, alpha=0.7, label='Calculated Centers (After)')

    # 绘制参考中心
    ax2.scatter(reference_center[0], reference_center[1], reference_center[2],
                c='blue', s=100, marker='*', label=reference_label)

    # 绘制连接线
    for point in after_points:
        ax2.plot([point[0], reference_center[0]], [point[1], reference_center[1]],
                 [point[2], reference_center[2]], 'g--', alpha=0.3)

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('After Calibration')
    ax2.legend()

    # 对比图
    ax3 = fig.add_subplot(133, projection='3d')

    # 绘制校准前后的对比
    ax3.scatter(before_points[:, 0], before_points[:, 1], before_points[:, 2],
                c='red', s=50, alpha=0.7, label='Before Calibration')
    ax3.scatter(after_points[:, 0], after_points[:, 1], after_points[:, 2],
                c='green', s=50, alpha=0.7, label='After Calibration')
    ax3.scatter(reference_center[0], reference_center[1], reference_center[2],
                c='blue', s=100, marker='*', label=reference_label)

    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title('Before vs After Calibration')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # 打印高精度统计信息
    print("\n=== High-Precision Calibration Statistics ===")
    
    if true_sphere_center is not None:
        print(f"Reference point: True sphere center")
    else:
        print(f"Reference point: Centroid of calibrated data")
        print(f"Reference center: [{reference_center[0]:.3f}, {reference_center[1]:.3f}, {reference_center[2]:.3f}] mm")

    # 计算校准前后的平均误差
    before_errors = [np.linalg.norm(center - reference_center) for center in sphere_centers_base_before]
    after_errors = [np.linalg.norm(center - reference_center) for center in sphere_centers_base_after]

    print(f"Before calibration:")
    print(f"  Mean distance from reference: {np.mean(before_errors):.3f} mm")
    print(f"  Std distance from reference: {np.std(before_errors):.3f} mm")
    print(f"  Max distance from reference: {np.max(before_errors):.3f} mm")
    print(f"  Min distance from reference: {np.min(before_errors):.3f} mm")

    print(f"\nAfter calibration:")
    print(f"  Mean distance from reference: {np.mean(after_errors):.8f} mm")
    print(f"  Std distance from reference: {np.std(after_errors):.8f} mm")
    print(f"  Max distance from reference: {np.max(after_errors):.8f} mm")
    print(f"  Min distance from reference: {np.min(after_errors):.8f} mm")
    
    # 计算分散度改善
    before_std = np.std(before_errors)
    after_std = np.std(after_errors)
    
    if before_std > 0:
        dispersion_improvement = ((before_std - after_std) / before_std * 100)
        print(f"\nDispersion improvement: {dispersion_improvement:.2f}%")
    
    # 计算距离改善百分比
    if np.mean(before_errors) > 0:
        improvement = ((np.mean(before_errors) - np.mean(after_errors)) / np.mean(before_errors) * 100)
        print(f"Mean distance improvement: {improvement:.2f}%")

    # 检查是否达到目标精度
    target_achieved = np.mean(after_errors) < target_precision
    print(f"Target precision ({target_precision:.8f}mm): {'Achieved' if target_achieved else 'Not achieved'}")

    # 显示超过目标精度的点的数量
    points_within_target = sum(1 for error in after_errors if error < target_precision)
    print(f"Points within target precision: {points_within_target}/{len(after_errors)} ({points_within_target / len(after_errors) * 100:.1f}%)")
    
def plot_optimization_history(calibration_results: Dict):
    """
    绘制优化过程的误差变化曲线

    :param calibration_results: 校准结果字典,包含优化历史和目标精度
    """
    # 从校准结果中提取数据
    optimization_history = calibration_results.get('optimization_history', [])
    target_precision = calibration_results.get('target_precision', 1.0)

    if len(optimization_history) == 0:
        print("Error: No optimization history available. Please provide valid calibration_results.")
        return

    plt.figure(figsize=(12, 8))

    # 创建子图
    plt.subplot(2, 1, 1)
    plt.plot(optimization_history, 'b-', linewidth=2)
    plt.axhline(y=target_precision, color='r', linestyle='--', label=f'Target: {target_precision:.1f} mm')
    plt.xlabel('Iteration')
    plt.ylabel('RMS Error (mm)')
    plt.title('Calibration Optimization History')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 对数坐标版本
    plt.subplot(2, 1, 2)
    plt.semilogy(optimization_history, 'b-', linewidth=2)
    plt.axhline(y=target_precision, color='r', linestyle='--', label=f'Target: {target_precision:.1f} mm')
    plt.xlabel('Iteration')
    plt.ylabel('RMS Error (mm) - Log Scale')
    plt.title('Calibration Optimization History (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()