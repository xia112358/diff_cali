"""Run the FR16 calibration and a separate observation-set comparison."""
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cali.calibrator import RobotCalibrator
from cali.pre_calibrator import PreCalibrator
from cali.robots import create_robot

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/fr16/generated"


def load(calibrator, path, limit=None):
    if not calibrator.load_observations_from_json(str(ROOT / path), limit):
        raise RuntimeError(f"Could not load {path}")


def validate(robot, observations):
    evaluator = RobotCalibrator(robot)
    evaluator.set_observations(observations)
    with torch.no_grad():
        variance = evaluator._compute_variance_error().item()
        centers = np.array([p.numpy() for p in evaluator._compute_sphere_centers_in_base()])
    return variance, centers


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    robot = create_robot("FR16")
    pre = PreCalibrator()
    load(pre, "data/hand_eye_data/pose2_test1/observations.json")
    initial = pre.pre_calibrator()
    if initial is None:
        raise RuntimeError("Hand-eye initialization failed")
    with torch.no_grad():
        for name, value in zip(("tx", "ty", "tz"), initial["translation"]):
            getattr(robot, name).fill_(float(value))
        for name, value in zip(("rx", "ry", "rz"), initial["euler_angles"]):
            getattr(robot, name).fill_(float(value))
    before = copy.deepcopy(robot)

    calibrator = RobotCalibrator(
        robot,
        calibration_params=['a', 'alpha', 'd', 'beta', 'offset', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'],
        excluded_params={0: ['d', 'beta'], 1: ['d'], 2: ['d'],
                         3: ['d', 'beta'], 4: ['beta'],
                         5: ['a', 'alpha', 'd', 'beta', 'offset']},
        learning_rate=0.01, weight_decay=1e-5, max_iterations=2000,
        gradient_clip_norm=1.0, target_precision=1e-4,
        convergence_tolerance=1e-6, enable_outlier_detection=False,
        log_interval=500, verbose=True,
    )
    load(calibrator, "data/main_data/pose3/observations.json")
    result = calibrator.calibrate()
    validation = RobotCalibrator(robot)
    load(validation, "data/main_data/pose4/observations.json", 30)
    variance_before, centers_before = validate(before, validation.observations)
    variance_after, centers_after = validate(robot, validation.observations)
    print(f"Validation variance (mm²): {variance_before:.6f} -> {variance_after:.6f}")
    print(f"Validation sqrt(total variance) (mm): {np.sqrt(variance_before):.3f} -> {np.sqrt(variance_after):.3f}")
    print(f"Training target achieved: {result['precision_achieved']}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), subplot_kw={"projection": "3d"})
    for ax, points, label, var in zip(axes, (centers_before, centers_after),
                                      ("Before", "After"), (variance_before, variance_after)):
        ax.scatter(*points.T, s=20)
        ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
               title=f"{label}: variance {var:.4f} mm²")
    fig.tight_layout()
    fig.savefig(OUTPUT / "validation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.semilogy(result['optimization_history'], label="Training variance")
    ax.axhline(result['target_precision'], color="red", linestyle="--", label="Target")
    ax.set(xlabel="Iteration", ylabel="Sum of sample variances (mm²)", title="FR16 optimization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "optimization.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
