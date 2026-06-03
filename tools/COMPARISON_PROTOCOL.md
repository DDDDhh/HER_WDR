# Reproducing the paper's experiment section

Two things need to run on the server:

1. **Out-of-distribution evaluation** of the already-trained HER-WDR checkpoint.
   Fast (~2 min per task × 4 tasks = ~10 min on a beefy box).
2. **Baseline training**: vanilla PPO and naive HER, so we can replace the
   synthetic comparison figures with real TensorBoard curves.
   Long (~14 h per config on a 64-core CPU server).

Run them in this order. After each step, send me the terminal log or the
output file path and I will update `main.tex` and the figures.

---

## Step 1 — OOD evaluation of the trained HER-WDR checkpoint

Already-trained checkpoint:
```
data/logs/2026-05-11_21-59-36/model/best_model.pth
```

**Launch:**

```bash
cd parafoil_train_basic
bash tools/run_ood_eval.sh \
    -m data/logs/2026-05-11_21-59-36/model/best_model.pth \
    -n 200 -w 25
```

This sequentially evaluates the checkpoint on four task variants:

| Tag             | Task yml                                 | What it tests                                      |
|-----------------|------------------------------------------|----------------------------------------------------|
| in_distribution | `flare_landing.yml`                      | Sanity re-check at seed 42 (different from your 100 % run) |
| ood_strong_wind | `flare_landing_ood_strong.yml`           | Wind speed 5–8 m/s (training was 1–4 m/s)          |
| ood_gusty       | `flare_landing_ood_gusty.yml`            | 3× gust + turbulence amplitude, mean wind in-range |
| ood_combined    | `flare_landing_ood_combined.yml`         | Strong mean + amplified gusts                      |

**Check results:** Combined log is saved to
`data/eval_ood_<timestamp>.txt`. The interesting lines are the `Success
(complete)` / `Avg Heading Align` / `Avg Final Flare` blocks at the bottom of
each task section.

**Send back:** Just paste me the contents of the `data/eval_ood_<timestamp>.txt`
file (or attach it). I will fold the four numbers into `main.tex` and add the
OOD-robustness subsection.

---

## Step 2 — Train the two baselines

The HER-WDR run already exists. We still need vanilla PPO and naive HER.

**Launch both in detached tmux sessions:**

```bash
cd parafoil_train_basic
bash tools/start_train.sh -s vanilla_ppo -c config/framework_ppo_baseline.yml
bash tools/start_train.sh -s her_naive   -c config/framework_ppo_her_standard.yml
```

(There is also `bash tools/run_comparison.sh` which kicks off all three at
once, but you already have the HER-WDR run — so just launch the two
baselines individually.)

**Check progress:**

```bash
tmux ls                          # see the two sessions
tmux attach -t vanilla_ppo       # peek at the rolling logs; Ctrl-b d to detach
tmux attach -t her_naive
```

Each run writes its log dir to `data/logs/<timestamp>/`. The most useful
single-number progress check is the TensorBoard scalar dump:

```bash
tensorboard --logdir data/runs --port 6006 --bind_all
# then open http://<server-ip>:6006
```

The runs are configured for `max_episode: 200000` but in practice convergence
is reached by ~30–50k episodes. You can let them go to completion (~14 h
each) or stop them after the curves flatten.

---

## Step 3 — Eval the two baselines on the same protocol

After each baseline finishes, evaluate its best checkpoint on the
in-distribution task (and optionally the OOD tasks):

```bash
# Vanilla PPO
python tools/evaluate_model.py \
    --model data/logs/<vanilla_ppo_timestamp>/model/best_model.pth \
    --episodes 200 --workers 25 --seed 42

# Naive HER
python tools/evaluate_model.py \
    --model data/logs/<her_naive_timestamp>/model/best_model.pth \
    --episodes 200 --workers 25 --seed 42
```

Send me the two terminal outputs.

---

## Step 4 — Comparison plots from TensorBoard event files

Once all three trainings have written event files (you can do this even while
the baselines are still going, but the curves will be short):

```bash
python tools/plot_comparison.py \
    --run her_wdr=data/runs/<her_wdr_run_name> \
    --run vanilla_ppo=data/runs/<vanilla_ppo_run_name> \
    --run her_naive=data/runs/<her_naive_run_name> \
    --out paper_figs/
```

Find the run names with `ls data/runs/`. (They are whatever the training
script picked — usually `<timestamp>__<host>` or the tmux session name.)

This writes one PNG per scalar tag (reward, real success rate, HER rate,
avg step, dir_err) plus a `summary.txt` file with end-of-training means.

Send me the contents of `paper_figs/summary.txt` and I will:

1. Replace the synthetic figures in `paper/Format/` with the real PNGs from
   `paper_figs/`.
2. Update Tables `tab:comparison`, `tab:ablation-her`, `tab:ablation-reward`
   in `main.tex` with the real numbers.

---

## Quick troubleshooting

* `OSError: [WinError 1455] page file too small` → reduce `--workers`.
  25 is safe on a 64 GB box; bump to 50 only when no training is running.
* `Cannot load mkl_intel_thread.2.dll` → same root cause; conda Python is
  out of memory for child processes.
* `tmux: command not found` → install tmux or run the inner command
  directly:
  ```bash
  export FRAMEWORK_CONFIG_PATH=$PWD/config/framework_ppo_baseline.yml
  python framework/framework_train.py
  ```
