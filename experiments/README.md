# Experiments

The four experiments reported in the paper, with the configuration each was run
at. All physics-informed runs are on benchmark B2 at `nt = 50`, which is the
configuration reported throughout and fits in 7.8 GB of GPU memory at the
largest collocation density.

Commands below are given from `src/pinn/b2/`. Set `GW_DATA` first; see
[`../src/paths.py`](../src/paths.py).

## Shared physics-informed configuration

| Setting | Value |
|---|---|
| Network | 5 hidden layers x 50 neurons, sine activation, learnable scale |
| Constraint | `HARD` (Dirichlet and initial conditions by construction) |
| Stages | 1: t in [0, tau]; 2: t in [tau, 30], tau = 1 d |
| Optimiser | Adam 4000 epochs, then L-BFGS 1000 epochs |
| Well kernel | Gaussian, sigma = 30 m |
| Sampling | spatial `LR` (locally refined), temporal `LHS` |
| Loss weight | `--alpha_fixed 0.05` |
| L-BFGS | line search on, `--lbfgs_max_iter 20`, `--lbfgs_div_retries 3` |
| Seed | 200 |
| Split | chronological (`--chrono`): train to day 20, validate 21-25, test 26-30 |

`--lbfgs_div_retries` sets how many times the L-BFGS phase may restore weights
and retry after the validation loss exceeds 1.2x its best. A noisy objective
crosses that threshold far more often than a clean one, so the noise arms
exhaust a small budget long before the epoch budget. Raising it affects only how
many restarts are permitted; `--lbfgs_max_iter` is unchanged, so no arm receives
more optimisation per epoch than the baseline.

Common flags:

```bash
COMMON="--chrono --spatial_strategy LR --temporal_strategy LHS --nt 50 \
        --tau 1 --sigma 30 --constraint HARD \
        --epochs_Adam 4000 --epochs_LBFGS 1000 \
        --alpha_fixed 0.05 --lbfgs_line_search --lbfgs_div_retries 3 \
        --field_pattern '$GW_DATA/benchmarks/b2/t*.txt'"
```

Run stage 1, then stage 2, for each arm:

```bash
python train.py --stage 1 $COMMON --filename <cloud>.mat --anchor_pattern <anchors> --run_tag <tag>
python train.py --stage 2 $COMMON --filename <cloud>.mat --anchor_pattern <anchors> --run_tag <tag>
```

Evaluate with the B2 evaluator in `direct` mode:

```bash
python ../../diagnostics/evaluate_b2.py \
    --prediction-dir <run>/test_predictions \
    --output-dir     <out> \
    --model-name     "PINN B2 <tag>" \
    --prediction-mode direct
```

## 1. Collocation density

Four interior point counts at a fixed 815 supervised anchors per training day.
This is the knob that moves conservation without moving accuracy much.

| Arm | Interior points | Anchors/day | Collocation points at nt = 50 |
|---|---:|---:|---:|
| C1 | 1,057 | 815 | 52,850 |
| C2 | 1,976 | 815 | 98,800 |
| C3 | 3,803 | 815 | 190,150 |
| C4 | 5,629 | 815 | 281,450 |

C3 is the reported arm: it is the default cloud at the full anchor set, so it
serves as the collocation sweep's baseline and the supervision sweep's
815-anchor case.

## 2. Supervision density

Three anchor counts at the fixed C3 cloud of 3,803 interior points: 204, 408 and
815 anchors per training day. Anchor subsets are drawn with seed 200.

## 3. Observation noise

Gaussian noise added to the training anchors only. Validation and test targets
stay clean, and the prescribed-head rows are never perturbed. Noise levels are
expressed as a fraction of the maximum drawdown, `Dmax = 9.09 m`.

| Arm | Noise | sigma |
|---|---|---:|
| clean | 0% | 0 m |
| noise05 | 5% | 0.4392 m |
| noise10 | 10% | 0.8784 m |

The clean arm is C3, which is the same cloud with the same 815 clean anchors, so
the sweep is one configuration throughout.

Check L-BFGS coverage before comparing noise arms: an arm stopped early by the
divergence guard is undertrained, and undertraining degrades the conservation
diagnostics in the same direction as noise. Report the logged L-BFGS epochs
alongside the result.

## 4. Data-driven ablations

Two scripts, both benchmark-parameterised and runnable directly:

```bash
python receptive_field_ablation/run_receptive_field_ablation.py --benchmark both --diagnostics
python architecture_study/run_convlstm_architecture_study.py    --benchmark both --variant dilated_seq3
```

`collect_ablation_results.py` and `collect_convlstm_architecture_results.py`
gather the per-arm exports into one table each.

The reported CNN is `rf127_c48_d1-2-4-8-16-32`; the reported ConvLSTM is
`dilated_seq3_h32_k5_d4_l1`.
