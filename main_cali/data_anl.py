from typing import Dict
import numpy as np      
import matplotlib.pyplot as plt






def plot_calibration_results(calibration_results: Dict, true_sphere_center=None):
    """
    绘制校准前后的点云中心在基坐标系下的三维图

    :param calibration_results: 校准结果字典,包含校准前后的点云中心和误差信息
    :param true_sphere_center: 真实球心位置 [x, y, z]，如果不提供则从calibration_results中获取
    """
    # 从校准结果中提取数据
    sphere_centers_base_before = calibration_results.get('sphere_centers_base_before', [])
    sphere_centers_base_after = calibration_results.get('sphere_centers_base_after', [])
    target_precision = calibration_results.get('target_precision', 1.0)
    
    # 获取真实球心位置
    if true_sphere_center is not None:
        sphere_center_base = true_sphere_center
    else:
        sphere_center_base = calibration_results.get('sphere_center_base', None)

    if not sphere_centers_base_before or not sphere_centers_base_after or sphere_center_base is None:
        print("Error: Missing calibration data or true sphere center. Please provide valid calibration_results and/or true_sphere_center.")
        return

    # 创建3D图
    fig = plt.figure(figsize=(15, 5))

    # 校准前的图
    ax1 = fig.add_subplot(131, projection='3d')

    # 转换为numpy数组用于绘图
    before_points = np.array([center for center in sphere_centers_base_before])
    true_center = np.array(sphere_center_base)

    # 绘制校准前的点
    ax1.scatter(before_points[:, 0], before_points[:, 1], before_points[:, 2],
                c='red', s=50, alpha=0.7, label='Calculated Centers (Before)')

    # 绘制真实球心
    ax1.scatter(true_center[0], true_center[1], true_center[2],
                c='blue', s=100, marker='*', label='True Center')

    # 绘制连接线
    for point in before_points:
        ax1.plot([point[0], true_center[0]], [point[1], true_center[1]],
                 [point[2], true_center[2]], 'r--', alpha=0.3)

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

    # 绘制真实球心
    ax2.scatter(true_center[0], true_center[1], true_center[2],
                c='blue', s=100, marker='*', label='True Center')

    # 绘制连接线
    for point in after_points:
        ax2.plot([point[0], true_center[0]], [point[1], true_center[1]],
                 [point[2], true_center[2]], 'g--', alpha=0.3)

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
    ax3.scatter(true_center[0], true_center[1], true_center[2],
                c='blue', s=100, marker='*', label='True Center')

    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title('Before vs After Calibration')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # 打印高精度统计信息
    print("\n=== High-Precision Calibration Statistics ===")

    # 计算校准前后的平均误差
    before_errors = [np.linalg.norm(center - true_center) for center in sphere_centers_base_before]
    after_errors = [np.linalg.norm(center - true_center) for center in sphere_centers_base_after]

    print(f"Before calibration:")
    print(f"  Mean error: {np.mean(before_errors):.3f} mm")
    print(f"  Std error: {np.std(before_errors):.3f} mm")
    print(f"  Max error: {np.max(before_errors):.3f} mm")
    print(f"  Min error: {np.min(before_errors):.3f} mm")

    print(f"\nAfter calibration:")
    print(f"  Mean error: {np.mean(after_errors):.8f} mm")
    print(f"  Std error: {np.std(after_errors):.8f} mm")
    print(f"  Max error: {np.max(after_errors):.8f} mm")
    print(f"  Min error: {np.min(after_errors):.8f} mm")

    # 计算改善百分比
    improvement = ((np.mean(before_errors) - np.mean(after_errors)) / np.mean(before_errors) * 100)
    print(f"\nMean error improvement: {improvement:.8f}%")

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