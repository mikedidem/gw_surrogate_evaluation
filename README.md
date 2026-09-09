# Conservation Diagnostics for Deep Learning Groundwater Surrogates

Companion code for:

> **Beyond Accuracy: Conservation Diagnostics for Deep Learning Groundwater
> Surrogates.** Michael Edidem, Ruopu Li, Pouria Kharazi. Manuscript under
> review.

A deep-learning surrogate can reproduce hydraulic head to R² > 0.999 and still
violate the physics it was trained on. This repository implements a diagnostic
framework that evaluates predictive accuracy jointly with governing-equation
residuals, boundary-condition compliance, boundary-gradient direction and
mass-balance closure, and applies it to three surrogates on two benchmarks.

---

## The result in one table

Unconstrained arms over the test window t = 26-30 d, seed 42 for the data-driven
models and seed 200 for the physics-informed model. The CNN and ConvLSTM are
evaluated single-step; the physics-informed model is a direct spatiotemporal
query, so no stepping is involved. Head RMSE and the residual ratio are means
over the window, and mass error is the mean absolute residual over it as a
percentage of the imposed extraction. Every number is produced by the code in
this repository; see [Reproducing a result](#reproducing-a-result).

| Benchmark | Model | Head RMSE (m) | R² | PDE residual ratio | Mass error (% of well) |
|---|---|---:|---:|---:|---:|
| B1 | CNN | 0.0144 | 0.99985 | 84.0 | 37.8 |
| B1 | ConvLSTM | 0.0232 | 0.99960 | 88.6 | 289.6 |
| B1 | PINN | 0.0092 | 0.99994 | **0.13** | **0.42** |
| B2 | CNN | 0.0366 | 0.99989 | 137.9 | 62.4 |
| B2 | ConvLSTM | 0.0652 | 0.99964 | 289.1 | 114.0 |
| B2 | PINN | 0.0147 | 0.99998 | **0.51** | **1.5** |

The residual ratio is the model's governing-equation residual divided by
MODFLOW's own residual under the same discrete operator, so a value of 1 means
the surrogate is as consistent as the solver that generated its training data.
All six models pass any conventional accuracy screen. Three of them carry
residuals two orders of magnitude above the numerical baseline and fail to close
mass balance.

![Residual statistics](docs/images/residual_statistics.png)

---

## The two benchmarks

![Benchmark domain configurations](docs/images/benchmarks.png)

Both are transient, unconfined, single-layer aquifers on a 1000 × 1000 m domain
discretised at 300 × 300, simulated in MODFLOW-2005 over 0–30 days with 33
saved snapshots. Dirichlet conditions apply on the north and south edges,
no-flow east and west.

| | B1 | B2 |
|---|---|---|
| Hydraulic conductivity | 33.33 m/d, uniform | 33.33 m/d with a 10:1 low-K lens |
| Specific yield | 0.10 | 0.10 |
| Boundary head, south / north | 90 m / 90 m | 90 m / 100 m |
| Initial head | 90 m, uniform | Dupuit profile for the regional gradient |
| Well | Gaussian, σ = 30 m, centred | Gaussian, σ = 30 m, centred |
| Pumping rate | −40,000 m³/d | −40,000 m³/d |

The B2 lens sits between the well and the upgradient boundary, so it throttles
the well's principal supply. Because K varies in space, the Boussinesq operator
gains a term $h\,(\nabla K \cdot \nabla h)$; `problem.py` returns K together
with its analytic gradient for that reason.

---

## The three surrogates

![Surrogate architectures](docs/images/architectures.png)

| Model | Configuration | Prediction mode |
|---|---|---|
| CNN | six dilated 3×3 convolutions, 48 channels, dilations 1–32, GELU, receptive field 127 cells (~423 m), 104,449 parameters | single-step and autoregressive rollout |
| ConvLSTM | dilated, sequence length 3, 32 hidden channels, 5×5 kernel, dilation 4, one layer, receptive field 23–55 cells | single-step and autoregressive rollout |
| PINN | 5 hidden layers × 50 neurons, sine activation with a learnable per-layer scale, residual skip connections, two-stage Adam → L-BFGS | direct spatiotemporal query |

The physics-informed model has no single-step/rollout distinction: it is queried
at (x, y, t) directly, so there is nothing to roll out.

Its Dirichlet and initial conditions hold identically by construction, through a
multiplicative trial function:

$$
d = \frac{(t - t_{\min})(y_{\max} - y)(y - y_{\min})}
         {(t_{\max} - t_{\min})(y_{\max} - y_{\min})^2}
\qquad\qquad
h = h^{*} + d \cdot u
$$

where $d$ vanishes at $t = t_{\min}$ and at both $y$ boundaries whatever the
network outputs $u$. `src/constrained.py` applies the gridded analogue of the
same idea to the CNN and ConvLSTM, so that boundary enforcement can be tested
independently of the PDE residual term.

---

## The diagnostics

`src/diagnostics/groundwater_diagnostics.py` computes, per snapshot and pooled:

| Diagnostic | Meaning |
|---|---|
| Accuracy | MSE, MAE, RMSE, relative RMSE, R², all in metres |
| PDE residual | pointwise Boussinesq violation, and its ratio to MODFLOW's residual under the same stencil |
| Boundary error ε_BE | head discrepancy at the common boundary-evaluation rows |
| Boundary gradient | sign agreement with the reference at each Dirichlet boundary |
| Mass balance | storage change against boundary and well fluxes, as a percentage of the imposed extraction |
| Zone analysis | the same quantities split into well-influence, lens and background zones |
| Grid sensitivity | every residual re-evaluated on coarser stencils |

Predictions must be supplied as **absolute head in metres**; normalisation is
reversed before the diagnostics run, and normalised model losses are never
accepted as diagnostic input. Units are carried in the column names: accuracy in
metres, residuals in `_m_per_d`, fluxes and storage in `_m3_per_d`.

Each benchmark carries its own configuration, and both evaluators reject
reference data that does not match:

```bash
python src/diagnostics/evaluate_b2.py \
    --prediction-dir  "$GW_DATA/predictions/…/test_predictions" \
    --output-dir      out/b2_cnn \
    --model-name      "CNN B2 unconstrained" \
    --prediction-mode one-step
```

---

## Repository layout

```
src/
  pinn/b1/, pinn/b2/     two-stage physics-informed trainer, one tree per
                         benchmark; problem.py is the only file that differs
  cnn/                   dilated CNN surrogate
  convlstm/              temporal-residual ConvLSTM surrogate
  diagnostics/           the diagnostic package and the two evaluators
  constrained.py         hard Dirichlet enforcement for the gridded models
  paths.py               data-root resolution, used everywhere

experiments/             noise sweep, collocation and supervision sweeps,
                         receptive-field ablation, architecture study
tests/                   diagnostics self-checks
docs/                    images embedded above
```

B1 and B2 keep separate PINN trees deliberately. They differ only in
`problem.py`, but each writes its own `checkpoints/`, so the two cannot collide
and neither can silently inherit the other's configuration.

---

## Installation

```bash
git clone https://github.com/mikedidem/gw_surrogate_evaluation.git
cd gw_surrogate_evaluation
pip install -r requirements.txt
```

Training was run on Python 3.13 with PyTorch 2.11.0+cu128, CUDA 12.8 and NumPy
2.1.3, on a single Tesla T4.

---

## Data availability

This repository holds **code only**. The MODFLOW models, exported head
snapshots, trained checkpoints and diagnostic exports run to several gigabytes
and are distributed separately.

Point the `GW_DATA` environment variable at wherever that data lives locally:

```bash
export GW_DATA=/path/to/gw_surrogate_data      # Linux, macOS
set     GW_DATA=D:\gw_surrogate_data           # Windows
```

If `GW_DATA` is unset, `./data` is used instead, so a local copy or symlink
placed there needs no configuration. `src/paths.py` is the single place any of
this is resolved; nothing else in the codebase hard-codes a location. The
expected layout:

| Path under `GW_DATA` | Contents |
|---|---|
| `benchmarks/b1/`, `benchmarks/b2/` | MODFLOW models and `t*.txt` head snapshots |
| `diagnostic_results/` | one export per evaluated arm |
| `predictions/` | surrogate head fields, in absolute metres |

## Training a model

All three surrogates read benchmark data from `GW_DATA`; see [Data
availability](#data-availability). Each command below trains the reported
configuration.

### PINN

`src/pinn/b1/` and `src/pinn/b2/` are separate, self-contained trees -- the
benchmark is selected by which directory you run from, not by a flag, so a B1
run can never pick up a B2 configuration by accident. Training is two stages:
stage 1 covers the short initial window, stage 2 covers the full horizon and
resumes from stage 1's checkpoint.

A quick check that the pipeline runs, with points sampled on the fly rather
than from a precomputed collocation cloud:

```bash
cd src/pinn/b2
python train.py --stage 1 --chrono --constraint HARD \
    --spatial_strategy UNIFORM --nx 40 --ny 40 \
    --temporal_strategy LHS --nt 10 --tau 1 --sigma 30 \
    --anchor_pattern "$GW_DATA/benchmarks/b2/sdata/t*.txt" \
    --field_pattern  "$GW_DATA/benchmarks/b2/t*.txt" \
    --epochs_Adam 5 --epochs_LBFGS 0 --alpha_fixed 0.05
```

The reported arms use `--spatial_strategy LR` instead, which draws from a
precomputed locally-refined collocation cloud (`--filename`, a `.mat` file)
distributed with the rest of the data, at the full `nt = 50` and epoch budget:

```bash
python train.py --stage 1 --chrono --constraint HARD \
    --spatial_strategy LR --filename "$GW_DATA/<collocation cloud>.mat" \
    --temporal_strategy LHS --nt 50 --tau 1 --sigma 30 \
    --anchor_pattern "$GW_DATA/benchmarks/b2/sdata/t*.txt" \
    --field_pattern  "$GW_DATA/benchmarks/b2/t*.txt" \
    --epochs_Adam 4000 --epochs_LBFGS 1000 \
    --alpha_fixed 0.05 --lbfgs_line_search --lbfgs_div_retries 3

python train.py --stage 2 --chrono --constraint HARD \
    --spatial_strategy LR --filename "$GW_DATA/<collocation cloud>.mat" \
    --temporal_strategy LHS --nt 50 --tau 1 --sigma 30 \
    --anchor_pattern "$GW_DATA/benchmarks/b2/sdata/t*.txt" \
    --field_pattern  "$GW_DATA/benchmarks/b2/t*.txt" \
    --epochs_Adam 4000 --epochs_LBFGS 1000 \
    --alpha_fixed 0.05 --lbfgs_line_search --lbfgs_div_retries 3

python test.py --stage 2 --constraint HARD --spatial_strategy LR --sigma 30
```

See [`experiments/README.md`](experiments/README.md) for the collocation-density,
supervision-density and noise sweeps this trainer also runs, and for what each
flag controls.

### CNN

The reported architecture (C48-RF127: six dilated convolutions, 48 channels,
receptive field 127 cells) is trained through the receptive-field driver, which
wraps `src/cnn/cnn_surrogate.py` and controls its dilation schedule directly:

```bash
python experiments/receptive_field_ablation/run_receptive_field_ablation.py \
    --benchmark b2 --arm unconstrained \
    --dilations 1,2,4,8,16,32 --channels 48 --epochs 300 --diagnostics
```

`src/cnn/cnn_surrogate.py` is also runnable directly and is the canonical
implementation the driver imports; its own `--arch` flag exposes two fixed
configurations (`plain`, the original 3-layer, 7-cell model, and `dilated`, an
8-layer 511-cell model) rather than the six-layer 48-channel one reported in
the paper, which is why the driver is the way to reproduce that number
specifically.

### ConvLSTM

The reported architecture (dilated, sequence length 3, 32 hidden channels, 5x5
kernel, dilation 4, one recurrent layer) is the `dilated_seq3` variant:

```bash
python experiments/architecture_study/run_convlstm_architecture_study.py \
    --benchmark b2 --arm unconstrained --variant dilated_seq3 \
    --epochs 200 --diagnostics
```

Both drivers write `test_predictions/` (and `rollout_predictions/` for the
autoregressive arm) under their own `outputs/`, in the layout the diagnostics
evaluators expect directly -- no reformatting step between training and
evaluation.

---

## Reproducing a result

Two checks, cheapest first.

**Self-check.** Evaluating the reference data against itself must return zero
head error and a residual ratio of exactly one -- no surrogate involved, so this
only exercises the diagnostic pipeline itself:

```bash
export GW_DATA=/path/to/gw_surrogate_data
python -m pytest tests/ -q
```

It skips cleanly if the data tree isn't present.

**A table entry.** Run the matching evaluator on that arm's predictions and read
the result from `summary.json`. For the B2 CNN row above:

```bash
python src/diagnostics/evaluate_b2.py \
    --prediction-dir  "$GW_DATA/predictions/<model>/test_predictions" \
    --output-dir      out/b2_cnn \
    --model-name      "CNN B2 unconstrained" \
    --prediction-mode one-step
```

```
summary.accuracy.mean_rmse_m                      0.0366
summary.accuracy.mean_r2                          0.999888
summary.pde_residual.model_to_modflow_rmse_ratio  137.888
summary.mass_balance.mean_abs_percent_of_well     62.41
```

matching the B2 CNN row above. Every reported result is drawn from an export
this way, and each value is checked against the statistic its export recorded
before being reported, so nothing here can drift from the table it sits beside.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper rather than this
repository alone.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
