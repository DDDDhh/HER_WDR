# -*- coding: utf-8 -*-
"""
Read TensorBoard event files from multiple runs and produce comparison
figures suitable for the paper.

Usage:
  python tools/plot_comparison.py \
      --run her_ppo=data/runs/<her_run_name> \
      --run vanilla_ppo=data/runs/<vanilla_run_name> \
      --run her_naive=data/runs/<naive_run_name> \
      --out paper_figs/

For each requested scalar tag, draws one line per run with a rolling mean
and 1σ band. Tags default to the ones the logger emits.
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print('tensorboard package required. pip install tensorboard', file=sys.stderr)
    raise


DEFAULT_TAGS = [
    'RL/Episode Average Reward',
    'RL/real success rate',
    'RL/her rate',
    'RL/avg step',
    'RL/dir_err',
]
TAG_LABELS = {
    'RL/Episode Average Reward': 'Average episode reward',
    'RL/real success rate': 'Real success rate',
    'RL/her rate': 'HER relabel rate',
    'RL/avg step': 'Average episode length (RL steps)',
    'RL/dir_err': 'Final heading error (rad)',
}


def load_scalars(run_dir, tags):
    """Return ``{tag: (steps, values)}`` for each requested tag."""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    available = set(ea.Tags().get('scalars', []))
    out = {}
    for tag in tags:
        if tag not in available:
            print(f'  [warn] {run_dir} missing tag {tag!r}, skipping')
            continue
        events = ea.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in events], dtype=np.float64),
            np.array([e.value for e in events], dtype=np.float64),
        )
    return out


def rolling_mean(values, window):
    if len(values) < 2 or window <= 1:
        return values
    w = min(window, len(values))
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='valid')


def plot_tag(tag, runs, smoothing, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, scalars in runs.items():
        if tag not in scalars:
            continue
        steps, vals = scalars[tag]
        if smoothing > 1 and len(vals) > smoothing:
            smoothed = rolling_mean(vals, smoothing)
            offset = len(vals) - len(smoothed)
            ax.plot(steps[offset:], smoothed, label=name, linewidth=1.8)
            # std band on the smoothed signal
            std = np.array([
                np.std(vals[max(0, i - smoothing): i + 1])
                for i in range(offset, len(vals))
            ])
            ax.fill_between(steps[offset:], smoothed - std, smoothed + std, alpha=0.18)
        else:
            ax.plot(steps, vals, label=name, linewidth=1.8)

    ax.set_xlabel('Episodes')
    ax.set_ylabel(TAG_LABELS.get(tag, tag))
    ax.set_title(TAG_LABELS.get(tag, tag))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', framealpha=0.85)
    fig.tight_layout()
    safe = tag.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(out_dir, f'{safe}.png')
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True,
                        help='name=path; repeatable')
    parser.add_argument('--tag', action='append', default=None,
                        help='extra scalar tag to plot (repeatable)')
    parser.add_argument('--out', type=str, default='paper_figs')
    parser.add_argument('--smoothing', type=int, default=20,
                        help='rolling-mean window (in samples)')
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        if '=' not in item:
            print(f'expected NAME=PATH, got {item!r}', file=sys.stderr)
            sys.exit(2)
        name, path = item.split('=', 1)
        runs[name] = path

    tags = list(DEFAULT_TAGS)
    if args.tag:
        tags.extend(args.tag)

    os.makedirs(args.out, exist_ok=True)
    loaded = {name: load_scalars(path, tags) for name, path in runs.items()}

    for tag in tags:
        plot_tag(tag, loaded, args.smoothing, args.out)

    # Plain-text summary table of final-window means.
    summary_path = os.path.join(args.out, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('Run summary — mean of last 10% of recorded samples\n')
        f.write('=' * 60 + '\n')
        for name, scalars in loaded.items():
            f.write(f'\n[{name}]\n')
            for tag, (_, vals) in scalars.items():
                if len(vals) == 0:
                    continue
                tail = max(1, len(vals) // 10)
                mean = float(np.mean(vals[-tail:]))
                std = float(np.std(vals[-tail:]))
                f.write(f'  {tag:35s} {mean:9.4f} ± {std:.4f}  (n={tail})\n')
    print(f'  wrote {summary_path}')


if __name__ == '__main__':
    main()
