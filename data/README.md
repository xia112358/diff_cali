# Observation data

Each dataset is an `observations.json` file. The existing directory names are
preserved to keep paths used by the calibration code and FR16 example valid.

- `hand_eye_data/`: observations for hand-eye initialization.
- `main_data/`: observation sets used for calibration or validation; the FR16
  example calibrates on `pose3` and validates on the first 30 samples of `pose4`.
- `test_data/`: additional observation sets, not used by the FR16 example.

Robot presets use millimetres for lengths and radians for angles. Dataset names
alone do not establish independence or suitability for a particular robot.
