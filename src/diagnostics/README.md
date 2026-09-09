# Groundwater surrogate diagnostics

These are ordinary Python scripts, not an installable package. They demonstrate
how to evaluate groundwater surrogate predictions using both head accuracy and
physical-consistency metrics.

The diagnostic ideas are reusable, but each benchmark has its own physical
configuration:

| Setting | B1 | B2 |
|---|---:|---:|
| Hydraulic conductivity | constant 33.33 m/d | `kfield.txt`, approximately 10:1 contrast |
| South fixed head | 90 m | 90 m |
| North fixed head | 90 m | 100 m |
| Well | centered Gaussian, -40,000 m3/d | exact distributed rates from `b2_model.wel` |
| Temporal derivative | backward difference | NumPy gradient on saved times |
| PDE operator | expanded central difference | finite-volume `div(K h grad(h))` |

Both scripts reject reference data that do not match the selected benchmark.

## Metric units

Prediction files must contain absolute head in metres. Model normalization is
reversed before these diagnostics run; normalized model losses are not accepted
as diagnostic inputs or reported as accuracy.

Accuracy uses `mse_m2`, `mae_m`, `rmse_m`, `rrmse_percent`, and dimensionless
`r2`. PDE residual columns end in `_m_per_d`; flux and storage columns end in
`_m3_per_d`.

Normalised losses are internal optimisation quantities only. They are
dimensionless, they are not comparable between B1 and B2 because each
benchmark has its own training statistics, and they are never reported as
accuracy.

## Input files

The preferred format is one `x y h` text file per time:

```text
x y h
1.6667 1.6667 89.1234
5.0000 1.6667 89.1240
```

Plain 300 x 300 head matrices are also accepted. Matrix rows are assumed to
follow increasing y unless `--matrix-y-order descending` is supplied.

Temporal derivatives follow `--prediction-mode`. In `one-step` mode, every
predicted target is differenced from its observed reference predecessor. In
`rollout` mode, the first forecast is anchored to the observed predecessor and
later forecasts are differenced from the preceding prediction. In `direct` mode,
adjacent predicted fields define the trajectory derivative. For the standard
test, prediction files cover t26 through t30; the diagnostics obtain the day-25
reference anchor when needed. If a folder contains both `t26_pred.txt` and
`t26_gt.txt`, specify `--prediction-glob "*_pred.txt"`.

## Running

Reference data is located through `GW_DATA`; pass `--reference-dir` to override
it. Run from this directory:

```bash
python evaluate_b1.py \
  --prediction-dir  "$GW_DATA/predictions/<model>/test_predictions" \
  --output-dir      out/b1_cnn_one_step \
  --model-name      "CNN B1" \
  --prediction-mode one-step
```

For matrix-only ConvLSTM exports, where predictions and ground truth share a
directory:

```bash
python evaluate_b1.py \
  --prediction-dir  "$GW_DATA/predictions/<model>" \
  --prediction-glob "*_pred.txt" \
  --output-dir      out/b1_convlstm \
  --model-name      "ConvLSTM B1" \
  --prediction-mode one-step
```

Benchmark B2 uses the same interface:

```bash
python evaluate_b2.py \
  --prediction-dir  "$GW_DATA/predictions/<model>/test_predictions" \
  --output-dir      out/b2_cnn_one_step \
  --model-name      "CNN B2" \
  --prediction-mode one-step
```

One-step and autoregressive rollout predictions must use separate output
directories.

## Outputs

- `summary.json`: aggregate metrics and the full run configuration;
- `diagnostic_report.txt`: the complete console report with test, rollout, boundary, PDE, flux, and water-balance metrics;
- `accuracy_report.txt`: a compatibility copy of the complete printed report;
- `per_time_metrics.csv`: every metric at each evaluated time;
- `grid_sensitivity.csv`: residual sensitivity to block coarsening;
- `head_error_maps.png`: unclipped common-scale and per-time error maps;
- `pde_residual_maps.png`: surrogate and MODFLOW residuals on one scale;
- `diagnostic_timeseries.png`: accuracy, residual, flux, and mass balance.

The MODFLOW residual is always calculated using the same post-processing
operator as the surrogate. It is therefore reported as the numerical diagnostic
baseline rather than assumed to be zero.
