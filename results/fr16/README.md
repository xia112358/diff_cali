# FR16 example

The figures were generated with `python examples/fr16.py`: [optimization](optimization.png)
and [validation](validation.png). The example initializes hand-eye calibration
with `hand_eye_data/pose2_test1`, calibrates on `main_data/pose3`, and evaluates
the first 30 `main_data/pose4` observations.

In this run, the validation sum of sample variances decreased from 228.517406 to
0.090610 mm²; its square root decreased from 15.117 to 0.301 mm. The training
variance target of 0.0001 mm² was **not** reached. This metric measures spread
of the reconstructed center; it does not establish absolute positioning
accuracy or independence of the datasets. Results may vary between runs.
