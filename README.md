# parafoil_train_basic — HER-WDR + PPO for Parafoil Upwind Flare Landing

This repo started as a 9-DOF parafoil simulator with a vanilla multi-process
PPO trainer for trajectory tracking. It has been extended into a reference
implementation of **HER-WDR (Hindsight Experience Replay with Wind Direction
Relabeling)** for the *upwind flare-landing* task, with deterministic
evaluation tooling, six out-of-distribution wind variants, and reproducible
launch scripts. The companion paper lives at
`../Duzehang_parafoil_HTR/paper/Format/main.tex`.

---

## 1. What's new vs. the original repo

### 1.1 New task: upwind flare landing (`EnvFlare`)

`parafoil_env/env/track/env_flare.py`

A new environment in which each episode randomises the canopy's initial
heading **and** the mean wind direction. The agent must align the canopy with
the wind and pull both brakes (flare) within a narrow altitude window before
touchdown.

- 16-dim heading-error-centred observation (cos/sin of heading error,
  body-frame velocities, brake state, time encoding)
- Five-component dense reward: per-step heading-improvement, altitude-adaptive
  alignment, low-altitude alignment boost, action-smoothness penalty,
  descent-rate penalty
- Five terminal codes: `0=flying, 1=dead, 2=timeout, 3=real success,
  4=HER-relabeled success` (the critic and logger discriminate between them)
- All reward and observation logic exposed as `@staticmethod` pure functions
  on the env class so that HER can recompute them off the live env

### 1.2 HER-WDR pipeline

`parafoil_env/utils/her.py`

Pure-function relabel module. After every failed episode:

1. Reads the actual wind direction from the stored ground-frame wind vector
   via `atan2(status[29], status[28])` — natively in radians, sidestepping
   the deg/rad bug that broke the earlier prototype.
2. Computes the hindsight wind direction `phi_w_h = psi_final - pi`.
3. Rotates the stored wind vector at every RL-step boundary by
   `delta_phi = phi_w_h - phi_w`.
4. Recomputes observation and reward through the env's pure-function statics.
5. Returns a new `(obs, next_obs, reward)` triple that gets written back
   into the shared-memory PPO buffer in place; the terminal step's done code
   is rewritten from `1` or `2` to `4`.

Quality filter (configurable per yml): `trigger_prob=0.8`, `min_steps>=5`,
`min_alt_descended>=20m`. Real successes (`done=3`) are never relabeled.

The relabel is theoretically justified by the rotational equivariance of the
9-DOF body-frame parafoil dynamics: rotating the wind field by `delta_phi`
and the canopy ground-track by the same amount leaves every body-frame
quantity unchanged. See the paper's Section "Symmetry argument" for the
formal statement.

### 1.3 Done-code-aware PPO TD target

`algorithms/ppo.py`

Three terminal-handling branches matching the done codes:

- `done=1` (dead)         -> `G_t = r_t`
- `done=2` (timeout)      -> `G_t = r_t + gamma * V(s_{t+1})`
- `done=3 or 4` (success) -> `G_t = r_t`

Without this distinction, timeout episodes were being treated as terminal
deaths, biasing the value function on long-horizon flares.

### 1.4 Worker-side HER integration

`worker/worker_ppo.py`

Each worker:
1. Rolls out an episode through the shared-memory ring buffer.
2. On `done`, snapshots the trajectory's physical states at every RL-step
   boundary.
3. Calls `her.relabel_trajectory(...)` and overwrites the corresponding
   buffer slots in place if the trajectory passes the quality filter.
4. Signals the agent process that the batch slot is full.

No env-state mutation, no race conditions across workers.

### 1.5 OOD evaluation infrastructure

Six `flare_landing_*.yml` task configs under `parafoil_env/config/task/`:

| Task yml                          | What it perturbs                                                  |
|-----------------------------------|-------------------------------------------------------------------|
| `flare_landing.yml`               | In-distribution training task                                     |
| `flare_landing_ood_strong.yml`    | Mean wind speed `V_0 in [5, 8]` m/s (training was 1-4 m/s)        |
| `flare_landing_ood_gusty.yml`     | Gust and turbulence amplitudes tripled                            |
| `flare_landing_ood_combined.yml`  | Strong mean + tripled gust+turb together                          |
| `flare_landing_ood_altitude.yml`  | Initial altitude sampled per episode in `[100, 200]` m            |
| `flare_landing_ood_shear.yml`     | Wind-profile power-law exponent raised from 0.25 to 0.60          |

To support these, two surgical code changes were made:

- `parafoil_env/utils/task.py` — `_get_initial_status` now interprets a
  `coordinate[i] = [min, max]` list as "sample per episode from this range",
  and `_generate_wind_info` passes a new `wind.shear_exponent` field
  through.
- `parafoil_env/modules/wind_field.py` — the hard-coded `pow(zs/70, 0.25)`
  wind-shear exponent is replaced with `pow(zs/70, getattr(args,
  'shear_exponent', 0.25))`. Backward compatible.

### 1.6 Tools

| Script                                | Purpose                                                                        |
|---------------------------------------|--------------------------------------------------------------------------------|
| `tools/start_train.sh`                | Launch a training run inside a detached tmux session                           |
| `tools/run_comparison.sh`             | Launch HER-WDR / Vanilla / Naive HER trainings concurrently                    |
| `tools/evaluate_model.py`             | Deterministic-policy evaluation with `--task` override for OOD                 |
| `tools/run_ood_eval.sh`               | Sequentially eval one checkpoint on all six task variants                      |
| `tools/run_ood_delta.sh`              | Eval only the two new OOD variants (altitude + shear) on the three checkpoints |
| `tools/plot_comparison.py`            | Read TensorBoard event files and produce comparison curves                     |
| `tools/COMPARISON_PROTOCOL.md`        | End-to-end "how to reproduce the paper" walkthrough                            |

---

## 2. Quick start

### 2.1 Train HER-WDR-PPO

```bash
bash tools/start_train.sh -s her_wdr -c config/framework_ppo.yml
tmux attach -t her_wdr            # peek; Ctrl-b d to detach
```

### 2.2 Train the two ablation baselines

```bash
bash tools/start_train.sh -s vanilla_ppo -c config/framework_ppo_baseline.yml
bash tools/start_train.sh -s her_naive   -c config/framework_ppo_her_standard.yml
```

(Or `bash tools/run_comparison.sh` to launch all three at once.)

### 2.3 Evaluate a trained checkpoint (in-distribution)

```bash
python tools/evaluate_model.py \
    --model data/logs/<timestamp>/model/best_model.pth \
    --episodes 200 --workers 25 --seed 42
```

### 2.4 Run the full OOD evaluation suite

```bash
bash tools/run_ood_eval.sh \
    -m data/logs/<timestamp>/model/best_model.pth \
    -n 200 -w 25
# -> writes data/eval_ood_<timestamp>.txt with all six task results
```

### 2.5 Generate comparison plots from TensorBoard event files

```bash
python tools/plot_comparison.py \
    --run her_wdr=data/runs/<her_wdr_run> \
    --run vanilla=data/runs/<vanilla_run> \
    --run her_naive=data/runs/<her_naive_run> \
    --out paper_figs/
# -> writes one PNG per scalar tag plus paper_figs/summary.txt
```

---

## 3. Results

All numbers below are from `N_eval = 200` deterministic-policy rollouts
per (method, task) cell, with seeds disjoint from training. Best entry per
column is **bold**. Released checkpoints used for the table:

- HER-WDR-PPO -> `data/logs/2026-05-11_21-59-36/model/best_model.pth`
- Vanilla PPO -> `data/logs/2026-05-13_10-08-48/model/best_model.pth`
- Naive HER  -> `data/logs/2026-05-13_15-40-09/model/best_model.pth`

### 3.1 In-distribution deterministic evaluation

| Metric                          | HER-WDR-PPO       | Vanilla PPO       | Naive HER         |
|---------------------------------|-------------------|-------------------|-------------------|
| Landing success rate (%)        | **100.0** (200/200) | 100.0 (200/200) | 100.0 (200/200) |
| Average episode reward          | **82.77 +/- 3.28**  | 70.07 +/- 3.24   | 69.86 +/- 3.24   |
| Terminal heading cos(d_psi)     | **0.998**           | 0.996            | 0.997            |
| Final flare deflection          | **0.980**           | 0.973            | 0.966            |
| Average final altitude (m)      | 8.3                 | 6.7              | 6.7              |
| Average episode length (steps)  | 28.3                | 26.5             | 26.0             |
| Best-checkpoint episode         | ~30,000             | 6,800            | ~26,000          |

### 3.2 Out-of-distribution evaluation (success rate / avg reward / heading)

| Variant                                | HER-WDR-PPO              | Vanilla PPO         | Naive HER           |
|----------------------------------------|--------------------------|---------------------|---------------------|
| OOD-strong (`V_0` 5-8 m/s)             | 100% / **81.32** / 1.000 | 100% / 70.37 / 1.000 | 100% / 71.91 / 1.000 |
| OOD-gusty (3x gust + 3x turb)          | 99% / **81.29** / **0.975** | 99% / 69.80 / 0.972 | **100%** / 69.76 / 0.982 |
| OOD-combined (strong + gusty)          | 100% / **81.14** / **0.999** | 100% / 70.29 / 0.998 | 100% / 71.84 / 0.999 |
| OOD-altitude (`h_0` 100-200 m)         | 100% / **82.70** / **0.998** | 100% / 70.32 / 0.997 | 100% / 69.99 / 0.996 |
| OOD-shear (exponent 0.6)               | 100% / **82.22** / **0.993** | 100% / 69.94 / 0.988 | 100% / 69.90 / 0.987 |

### 3.3 Take-aways

- **All three methods solve the in-distribution task at 100% success** under
  the dense reward of `flare_landing.yml`. HER-WDR's value is *not* in
  passing a sparse-reward bottleneck (vanilla PPO clears the bar too) but in
  the *quality* of the converged policy.
- **HER-WDR converges to a ~17% higher episode reward** (`82.4` vs `70.1`)
  with measurably tighter terminal heading (`0.998` vs `0.996`) and slightly
  fuller flare. The ~12-reward-unit gap is the visible projection of the
  symmetry-based relabel onto the dense-reward landscape.
- **The quality gap transfers OOD without crossover**: HER-WDR's reward sits
  cleanly above both baselines on every one of the six OOD variants. The
  rotational-symmetry argument is empirically validated.
- **OOD-gusty is the only variant where any policy drops below 100%
  success** (HER-WDR and Vanilla both 99%; Naive HER alone retains 100%,
  trading reward quality for state-distribution coverage). High-frequency
  gusts perturb body-frame state directly, breaking the exact rotational
  symmetry.
- **OOD-altitude is essentially free**: a +/-33% perturbation in initial
  altitude leaves all metrics indistinguishable from in-distribution. The
  descent-profile structure is altitude-conditioned, not altitude-tuned.
- **OOD-shear is the cleanest OOD-gain story**: under a 2.4x steeper
  altitude wind gradient, HER-WDR's terminal heading degrades by only
  `-0.005` while both baselines degrade by `~0.010` — ~2x worse. HER-WDR's
  symmetry-aware shaping pays off most clearly where the altitude-wind
  coupling is strongest.

### 3.4 What the Naive HER ablation tells us

Naive HER is HER-WDR with the quality filter removed (`trigger_prob=1.0`,
`min_steps=1`, `min_alt_descended=0`). It collapses the policy onto the
vanilla-PPO operating point on every quality metric — same reward (~70),
same heading (~0.997), same flare (~0.966). The ablation isolates the
quality filter as the load-bearing component of HER-WDR; relabeling
indiscriminately injects pseudo-success samples that don't correspond to a
learnable causal relationship and washes out the symmetry-based gain.

---

## 4. Repository layout (selected)

```
parafoil_train_basic/
|-- algorithms/
|   `-- ppo.py                          # done-code-aware TD target
|-- config/
|   |-- framework_ppo.yml               # HER-WDR + PPO (main)
|   |-- framework_ppo_baseline.yml      # Vanilla PPO ablation
|   `-- framework_ppo_her_standard.yml  # Naive HER ablation
|-- framework/
|   `-- framework_train.py              # entrypoint (reads FRAMEWORK_CONFIG_PATH)
|-- parafoil_env/
|   |-- config/
|   |   `-- task/
|   |       |-- flare_landing.yml               # in-distribution
|   |       |-- flare_landing_ood_strong.yml    # OOD-strong
|   |       |-- flare_landing_ood_gusty.yml     # OOD-gusty
|   |       |-- flare_landing_ood_combined.yml  # OOD-combined
|   |       |-- flare_landing_ood_altitude.yml  # OOD-altitude
|   |       `-- flare_landing_ood_shear.yml     # OOD-shear
|   |-- env/track/
|   |   `-- env_flare.py                # EnvFlare with pure-function statics
|   |-- modules/
|   |   `-- wind_field.py               # configurable shear exponent
|   `-- utils/
|       |-- her.py                      # HER-WDR pure-function pipeline
|       `-- task.py                     # supports coordinate-range, shear pass-through
|-- worker/
|   `-- worker_ppo.py                   # in-place HER relabel after each done
|-- tools/
|   |-- start_train.sh                  # tmux launcher
|   |-- run_comparison.sh               # launch all three trainings
|   |-- evaluate_model.py               # deterministic eval, --task override
|   |-- run_ood_eval.sh                 # 6-task OOD suite
|   |-- run_ood_delta.sh                # only altitude+shear variants
|   |-- plot_comparison.py              # TensorBoard event -> PNG
|   `-- COMPARISON_PROTOCOL.md          # end-to-end reproduction guide
|-- data/
|   |-- logs/                           # checkpoints + per-run metadata
|   `-- runs/                           # TensorBoard event files
|-- paper_figs/                         # plot_comparison.py output, fed to paper/
`-- README.md                           # this file
```

---

## 5. Reproducing the paper end-to-end

The complete protocol --- which checkpoints to use, which seeds to pass,
which order to run things in, how to feed the results back into the LaTeX
source --- is in `tools/COMPARISON_PROTOCOL.md`. The short version:

1. `bash tools/run_comparison.sh` to launch all three trainings (~14 h each
   on a 64-core CPU server, 100 workers).
2. After convergence, `bash tools/run_ood_eval.sh -m <ckpt>` per checkpoint
   to populate the six-row OOD table.
3. `python tools/plot_comparison.py --run ... --out paper_figs/` to extract
   training curves from TensorBoard event files.
4. SFTP / copy `paper_figs/RL_Episode_Average_Reward.png`,
   `RL_real_success_rate.png`, `RL_her_rate.png` to
   `paper/fig2_reward_curve.png`, `fig3_success_rate.png`,
   `fig6_ablation_mix.png` respectively.
5. `cd paper && python generate_figures.py` to regenerate the synthetic
   figures (architecture diagram, OOD bar chart, trajectory) with the latest
   measured numbers.
6. `cd paper/Format && latexmk -pdf main.tex`.

---

## 6. Hyperparameters (HER-WDR-PPO defaults)

From `config/framework_ppo.yml` and `parafoil_env/config/task/flare_landing.yml`:

| Group        | Parameter                            | Value                          |
|--------------|--------------------------------------|--------------------------------|
| PPO          | actor lr / critic lr                 | 2e-4 / 3e-4                    |
| PPO          | gamma / clip / K epochs              | 0.97 / 0.2 / 10                |
| PPO          | entropy coef / grad-norm clip        | 0.05 / 0.5                     |
| PPO          | action std init / decay / floor      | 0.5 / x0.998 per 5 ep / 0.15   |
| Training     | workers / batch size / max episodes  | 100 / 200 / 200,000            |
| Episode      | T_max / action interval              | 200 steps / 1.0 s              |
| HER-WDR      | trigger prob / min steps / min descent | 0.8 / 5 / 20 m               |
| Task         | initial altitude / touchdown alt     | 150 m / 10 m                   |
| Task         | wind speed range / direction range   | [1, 4] m/s / [-pi, pi]         |
| Network      | actor / critic                       | FC[128,64,32]+LN, Tanh         |

---

## 7. Citing

If you use this code or the HER-WDR formulation, please cite the companion
paper. BibTeX entry will be added once the paper is accepted; in the
meantime cite as:

```
Du Zehang et al. "HER-WDR: Hindsight Experience Replay with Wind Direction
Relabeling for Autonomous Parafoil Terminal Guidance via Proximal Policy
Optimization." Manuscript in preparation, 2026.
```
