# HER-WDR: Hindsight Experience Replay with Wind Direction Relabeling

This repository implements **HER-WDR**, a novel Hindsight Experience Replay variant for autonomous parafoil upwind flare landing via Proximal Policy Optimization (PPO).

## Overview

HER-WDR addresses the sparse reward challenge in parafoil landing by retrospectively relabeling wind direction in failed episodes. The key insight is that parafoil dynamics exhibit rotational equivariance under wind field rotation, allowing physically consistent relabeling.

### Key Features

- **HER-WDR Algorithm**: Wind Direction Relabeling with quality filters
- **Done-code-aware PPO**: Custom TD target handling for death/timeout/success
- **Multi-process training**: Parallel environment workers with shared memory
- **OOD evaluation**: Six out-of-distribution wind variants for robustness testing

## Repository Structure

```
├── algorithms/
│   └── ppo.py                    # PPO with done-code-aware TD target
├── parafoil_env/
│   └── utils/
│       └── her.py                # HER-WDR core algorithm
├── framework/
│   └── framework_train.py        # Training entry point
├── worker/
│   ├── agent.py                  # PPO agent process
│   ├── worker_ppo.py             # Worker with HER integration
│   └── logger.py                 # TensorBoard logging
├── config/
│   ├── framework_ppo.yml         # HER-WDR + PPO (main)
│   ├── framework_ppo_baseline.yml # Vanilla PPO ablation
│   └── framework_ppo_her_standard.yml # Naive HER ablation
└── tools/
    ├── start_train.sh            # Launch training
    ├── evaluate_model.py         # Deterministic evaluation
    ├── run_ood_eval.sh           # OOD evaluation suite
    └── plot_comparison.py        # TensorBoard visualization
```

## Quick Start

### Training

```bash
# Train HER-WDR-PPO
bash tools/start_train.sh -s her_wdr -c config/framework_ppo.yml

# Train baselines
bash tools/start_train.sh -s vanilla_ppo -c config/framework_ppo_baseline.yml
bash tools/start_train.sh -s her_naive -c config/framework_ppo_her_standard.yml

# Or run all three concurrently
bash tools/run_comparison.sh
```

### Evaluation

```bash
# In-distribution evaluation
python tools/evaluate_model.py \
    --model data/logs/<timestamp>/model/best_model.pth \
    --episodes 200 --workers 25 --seed 42

# Full OOD evaluation suite
bash tools/run_ood_eval.sh \
    -m data/logs/<timestamp>/model/best_model.pth \
    -n 200 -w 25
```

## Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| PPO | actor lr / critic lr | 2e-4 / 3e-4 |
| PPO | gamma / clip / K epochs | 0.97 / 0.2 / 10 |
| PPO | entropy coef / grad-norm clip | 0.05 / 0.5 |
| HER-WDR | trigger prob / min steps / min descent | 0.8 / 5 / 20 m |
| Training | workers / batch size / max episodes | 100 / 200 / 200,000 |
| Network | actor / critic | FC[128,64,32]+LN, Tanh |

## Results

HER-WDR achieves:
- **84.5%** real landing success rate (vs 28.7% vanilla PPO, 58.7% naive HER)
- **12,500** episodes to reach 50% success threshold (vs not reached for baselines)
- Superior performance across all OOD wind variants

## Citation

If you use this code, please cite:

```bibtex
@article{du2026herwdr,
  title={HER-WDR: Hindsight Experience Replay with Wind Direction Relabeling for Autonomous Parafoil Terminal Guidance},
  author={Du, Zehang},
  year={2026}
}
```

## License

This code is released for academic research purposes.
