# Groundwater Surrogate Modeling — Evaluation Framework

Companion code for the paper: *"Accuracy Is Not Enough: A Physics-Informed Evaluation Framework
for Groundwater Surrogate Models"*

Three surrogate models are implemented and compared against MODFLOW:

| Model | Notebook (recommended) | Supporting scripts |
|---|---|---|
| CNN | `notebook/cnn.ipynb` | `gw_mod/cnn/` |
| ConvLSTM | `notebook/Convlstn.ipynb` | `gw_mod/` |
| GW-PINN | `notebook/PINN.ipynb` | `unconfined_m/` |

> **Start here:** Open the notebooks in `notebook/` to train, evaluate, and visualize all three models interactively.

---

## Problem Setup

- Domain: 1000 × 1000 m, single pumping well at (201.67, −98.33) m
- Aquifer type: unconfined, Boussinesq equation
- Parameters: K = 33.33 m/d, Sy = 0.10, Q = −40,000 m³/d
- Simulation period: 0–30 days (MODFLOW: 33 snapshots)

---

## Repository Structure

```
gw-surrogate-eval/
├── gw_mod/
│   └── cnn/                        # CNN surrogate
│       ├── cnn_modflow surogate_1.py   # Training and evaluation script
│       ├── utils.py                    # Metrics and plotting utilities
│       └── data/                       # MODFLOW snapshots (t0.25.txt … t30.txt)
├── unconfined_m/                   # GW-PINN (two-stage)
│   ├── trainer.py                  # Training script
│   ├── tester.py                   # Evaluation script
│   ├── model.py                    # Net, Net_Neumann, Net_PDE, PINN
│   ├── dataset.py                  # Trainset, Validset, ModflowDataset
│   ├── sampler.py                  # Spatial/temporal collocation samplers
│   ├── problem.py                  # PDE definition, BCs, source term
│   ├── options.py                  # CLI argument parser
│   └── utils.py                    # Plotting, metrics, checkpoint utilities
├── notebook/                       # *** START HERE — all three models ***
│   ├── cnn.ipynb                   # CNN: training, evaluation, and visualization
│   ├── Convlstn.ipynb              # ConvLSTM: training, evaluation, and visualization
│   └── PINN.ipynb                  # GW-PINN: two-stage training and evaluation
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data

MODFLOW snapshots are in `gw_mod/cnn/data/` as space-delimited text files:
```
x  y  head
...
```
Files are named `t{time}.txt` (e.g., `t0.25.txt`, `t1.0.txt`).

The PINN also requires a locally-refined mesh file `unconfined_m/data/well.mat`
(MATLAB format, generated with the mesh refinement procedure described in the paper).

---

## Running the Models

Open any notebook in Jupyter:

```bash
jupyter notebook notebook/cnn.ipynb          # CNN surrogate
jupyter notebook notebook/Convlstn.ipynb     # ConvLSTM surrogate
jupyter notebook notebook/PINN.ipynb         # GW-PINN (two-stage)
```

Each notebook covers data loading, model training, multi-step evaluation, and result visualization.

---

## Running the GW-PINN from the Command Line (alternative)

**Stage 1** (t = 0 → τ = 1 day):
```bash
cd unconfined_m/
python trainer.py --stage 1 --tau 1 --constraint HARD \
    --hidden_layers 5 --hidden_neurons 50 \
    --spatial_strategy LR --filename ./data/well.mat \
    --temporal_strategy LHS --nt 50 \
    --epochs_Adam 3000 --epochs_LBFGS 1000 \
    --sigma 30 --lam 100
```

**Stage 2** (t = τ → 30 days), requires a completed Stage 1 checkpoint:
```bash
python trainer.py --stage 2 --tau 1 --constraint HARD \
    --hidden_layers 5 --hidden_neurons 50 \
    --spatial_strategy LR --filename ./data/well.mat \
    --temporal_strategy LHS --nt 50 \
    --temporal_strategy_prev UNIFORM --nt_prev 50 \
    --epochs_Adam 3000 --epochs_LBFGS 1000 \
    --sigma 30 --lam 100
```

Checkpoints are saved under `unconfined_m/checkpoints/`.

---

## Evaluation Metrics

Beyond standard accuracy metrics (RMSE, MAE, R²), the framework computes:

- **PDE residual** — pointwise Boussinesq equation violation
- **Neumann boundary error** — flux conservation at no-flow boundaries
- **Mass balance error** — global water balance relative to MODFLOW
- **Zone-based analysis** — well-influence zone (r ≤ 90 m) vs. background zone

---


