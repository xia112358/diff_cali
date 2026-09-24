# diff_cali

A small robot kinematic calibration project implemented with PyTorch.

The project uses differentiable DH kinematics and gradient-based optimization to estimate robot kinematic parameters and hand-eye calibration parameters from repeated observations.

## Features

- Differentiable DH / beta-parameter robot model
- PyTorch autograd-based calibration
- Hand-eye pre-calibration
- Optional outlier rejection
- Calibration-result visualization
- Example observation datasets

Supported robot parameter presets currently include UR20, FR16, and ABB IRB 6700.

## Repository layout

```text
cali/
  calibrator.py       Main gradient-based calibrator
  pre_calibrator.py   Hand-eye pre-calibration
  mdh_robot.py        Differentiable robot kinematics
  robots.py           Robot parameter presets
  data_anl.py         Result visualization and analysis

data/
  test_data/          Example calibration observations
```

## Installation

```bash
pip install -e .
```

The main dependencies are PyTorch, NumPy, and Matplotlib.

## Basic usage

```python
from cali.robots import create_robot
from cali.calibrator import RobotCalibrator

robot = create_robot("FR16")
calibrator = RobotCalibrator(robot)
calibrator.load_observations_from_json("data/test_data/pose0/observations.json")
```

For hand-eye initialization, use `PreCalibrator` from `cali.pre_calibrator`.

## Notes

This is a compact engineering project rather than an actively maintained research codebase. Units in the bundled robot parameters and example observation data are millimetres unless noted otherwise.
