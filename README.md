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

## FR16 example and results

Run the former notebook workflow as a Python script from the repository root:

```bash
python -m pip install -e .
python examples/fr16.py
```

The example uses `data/hand_eye_data/pose2_test1` for hand-eye initialization,
`data/main_data/pose3` for calibration, and the first 30 observations in
`data/main_data/pose4` for validation. Generated plots are written to
`results/fr16/generated/`. Other datasets remain available under `data/`; their
names do not establish a train/test split. See [data notes](data/README.md).

The [FR16 results](results/fr16/README.md) include example plots and describe
their scope. These show dispersion on the recorded observations, not absolute
robot positioning accuracy.
