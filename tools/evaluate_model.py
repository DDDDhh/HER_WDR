# -*- coding: utf-8 -*-
"""
Evaluate a trained HER+PPO model on the upwind flare-landing task.

Usage:
  python tools/evaluate_model.py --model data/logs/<run>/model/best_model.pth \
      --episodes 200 --workers 25

Mirrors the analysis output the user is familiar with: per-episode line,
aggregate success/dead/timeout counts, mean reward / heading alignment /
final altitude / flare.
"""
import argparse
import contextlib
import io
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.append(package_root)

from algorithms.ppo import PPO
from parafoil_env.utils.config import Config
from parafoil_env.utils.task import Task
from parafoil_env.utils.method import angle_normalize


def _build_eval_config(model_path, task_override=None):
    framework_config_path = os.environ.get(
        'FRAMEWORK_CONFIG_PATH',
        os.path.join(package_root, 'config', 'framework_ppo.yml'),
    )
    framework_config = Config(framework_config_path).get_args()
    task_name = task_override or framework_config.proc.task_path
    if task_override:
        framework_config.proc.task_path = task_override
    task_config_path = os.path.join(
        package_root, 'parafoil_env', 'config', 'task', task_name
    )
    if not os.path.exists(task_config_path):
        raise FileNotFoundError(f'task config not found: {task_config_path}')
    task_config = Config(task_config_path).get_args()
    return framework_config, task_config


def _worker_run_episodes(args_tuple):
    model_path, num_episodes, seed, save_traj, worker_id, task_override = args_tuple
    framework_config, task_config = _build_eval_config(model_path, task_override)
    task_config.random_seed = task_config.random_seed + worker_id + seed

    task = Task(task_config)
    env_args = task.acquire_env_args()
    framework_config.proc.state_dim = env_args.observation.dim
    framework_config.proc.action_dim = env_args.action.dim
    framework_config.proc.max_episode_steps = env_args.max_episode_steps
    if hasattr(framework_config, 'agent'):
        framework_config.agent.device = 'cpu'

    with contextlib.redirect_stdout(io.StringIO()):
        ppo = PPO(framework_config)
        ppo.load(model_path)

    Env = env_args.env_class
    env = Env(env_args)

    results = {
        'complete': 0, 'dead': 0, 'done': 0,
        'rewards': [], 'steps': [],
        'final_altitudes': [], 'heading_alignments': [],
        'final_flare': [], 'min_altitudes': [],
    }
    ep_details = []
    all_trajectories = [] if save_traj else None

    for ep in range(num_episodes):
        env_args_i = task.acquire_env_args()
        state = env.reset(env_args_i)
        total_reward, step, min_alt = 0.0, 0, 999.0
        traj_states = [] if save_traj else None
        traj_actions = [] if save_traj else None

        while True:
            action, _ = ppo.select_action(state, sampling=False)
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            step += 1

            status = env.storage.status_now
            alt = -float(status[2])
            if alt < min_alt:
                min_alt = alt

            if save_traj:
                traj_states.append(np.copy(status))
                traj_actions.append(np.copy(action))

            if done:
                done_type = info.get('done_type', 'unknown')
                results[done_type] = results.get(done_type, 0) + 1
                results['rewards'].append(total_reward)
                results['steps'].append(step)

                altitude = -float(status[2])
                heading = float(status[8])
                flare = (float(status[22]) + float(status[24])) * 0.5
                # Use stored wind vector (radians) for consistency with the env.
                wx, wy = float(status[28]), float(status[29])
                wind_dir = float(np.arctan2(wy, wx)) if (abs(wx) + abs(wy) > 1e-8) else 0.0
                target_heading = wind_dir + np.pi
                err = float(abs(angle_normalize(heading - target_heading, unit='rad')))

                results['final_altitudes'].append(altitude)
                results['heading_alignments'].append(float(np.cos(err)))
                results['final_flare'].append(flare)
                results['min_altitudes'].append(min_alt)

                ep_details.append({
                    'worker_id': worker_id,
                    'ep': ep,
                    'done_type': done_type,
                    'reward': total_reward,
                    'steps': step,
                    'altitude': altitude,
                    'min_alt': min_alt,
                    'heading_err_deg': float(np.rad2deg(err)),
                    'flare': flare,
                })

                if save_traj:
                    all_trajectories.append({
                        'type': done_type,
                        'states': np.array(traj_states),
                        'actions': np.array(traj_actions),
                        'total_reward': total_reward,
                    })
                break

    env.close()
    try:
        env.smm.shm.unlink()
    except Exception:
        pass
    return results, ep_details, all_trajectories


def _merge(worker_outputs, save_traj):
    merged = {
        'complete': 0, 'dead': 0, 'done': 0,
        'rewards': [], 'steps': [],
        'final_altitudes': [], 'heading_alignments': [],
        'final_flare': [], 'min_altitudes': [],
    }
    all_details = []
    all_trajectories = [] if save_traj else None
    for results, ep_details, trajectories in worker_outputs:
        for k in ('complete', 'dead', 'done'):
            merged[k] += results[k]
        for k in ('rewards', 'steps', 'final_altitudes', 'heading_alignments',
                  'final_flare', 'min_altitudes'):
            merged[k].extend(results[k])
        all_details.extend(ep_details)
        if save_traj and trajectories:
            all_trajectories.extend(trajectories)
    all_details.sort(key=lambda x: (x['worker_id'], x['ep']))
    return merged, all_details, all_trajectories


def _print_results(results, n, seed, all_details, all_trajectories, save_traj):
    print('\n' + '=' * 70)
    print(f'  EVALUATION RESULTS  ({n} episodes, seed={seed})')
    print('=' * 70)
    for d in all_details:
        marker = ('✓' if d['done_type'] == 'complete'
                  else ('✗' if d['done_type'] == 'dead' else '⏱'))
        print(
            f"  {marker} Ep {d['ep']+1:3d} [W{d['worker_id']:02d}]: "
            f"{d['done_type']:8s} R={d['reward']:7.1f} steps={d['steps']:3d} "
            f"alt={d['altitude']:.1f}m min_alt={d['min_alt']:.1f}m "
            f"heading_err={d['heading_err_deg']:.1f}° flare={d['flare']:.2f}"
        )

    print(f"\n  Success (complete): {results['complete']:4d} "
          f"({results['complete']/n*100:5.1f}%)")
    print(f"  Dead:               {results['dead']:4d} "
          f"({results['dead']/n*100:5.1f}%)")
    print(f"  Timeout (done):     {results['done']:4d} "
          f"({results['done']/n*100:5.1f}%)")
    print(f"  {'-'*36}")
    print(f"  Avg Reward:         {np.mean(results['rewards']):8.2f} "
          f"± {np.std(results['rewards']):.2f}")
    print(f"  Avg Steps:          {np.mean(results['steps']):8.1f} "
          f"± {np.std(results['steps']):.1f}")
    print(f"  Avg Final Alt:      {np.mean(results['final_altitudes']):8.1f}m")
    print(f"  Avg Min Alt:        {np.mean(results['min_altitudes']):8.1f}m")
    print(f"  Avg Heading Align:  {np.mean(results['heading_alignments']):8.3f}"
          f"  (1.0 = perfect upwind)")
    print(f"  Avg Final Flare:    {np.mean(results['final_flare']):8.3f}")

    if results['complete'] > 0:
        success_rewards = [
            r for r, t in zip(results['rewards'],
                              [d['done_type'] for d in all_details])
            if t == 'complete'
        ]
        if success_rewards:
            print(f"\n  --- Success episodes ---")
            print(f"  Avg Reward (success): {np.mean(success_rewards):.2f}")

    if results['dead'] > 0:
        dead_alt = [
            a for a, t in zip(results['final_altitudes'],
                              [d['done_type'] for d in all_details])
            if t == 'dead'
        ]
        if dead_alt:
            print(f"  Avg Dead Final Alt:   {np.mean(dead_alt):.1f}m")

    print('=' * 70)
    return all_trajectories if (save_traj and all_trajectories) else None


def evaluate(model_path, num_episodes=100, render=False, seed=42, save_traj=False,
             num_workers=1, task_override=None):
    if num_workers > 1:
        return _evaluate_parallel(model_path, num_episodes, seed, save_traj,
                                  num_workers, task_override)
    return _evaluate_single(model_path, num_episodes, render, seed, save_traj,
                            task_override)


def _evaluate_single(model_path, num_episodes, render, seed, save_traj,
                     task_override=None):
    framework_config, task_config = _build_eval_config(model_path, task_override)
    task_config.random_seed = task_config.random_seed + seed
    task = Task(task_config)
    env_args = task.acquire_env_args()
    framework_config.proc.state_dim = env_args.observation.dim
    framework_config.proc.action_dim = env_args.action.dim
    framework_config.proc.max_episode_steps = env_args.max_episode_steps
    if hasattr(framework_config, 'agent'):
        framework_config.agent.device = 'cpu'

    ppo = PPO(framework_config)
    ppo.load(model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    print(f"Checkpoint episode: {ckpt.get('Episode', 'N/A')}")

    Env = env_args.env_class
    env = Env(env_args)
    if render:
        env.render()

    results = {
        'complete': 0, 'dead': 0, 'done': 0,
        'rewards': [], 'steps': [],
        'final_altitudes': [], 'heading_alignments': [],
        'final_flare': [], 'min_altitudes': [],
    }
    all_details = []
    all_trajectories = [] if save_traj else None

    for ep in range(num_episodes):
        env_args_i = task.acquire_env_args()
        state = env.reset(env_args_i)
        total_reward, step, min_alt = 0.0, 0, 999.0
        traj_states = [] if save_traj else None
        traj_actions = [] if save_traj else None
        while True:
            action, _ = ppo.select_action(state, sampling=False)
            state, reward, done, info = env.step(action)
            total_reward += float(reward)
            step += 1
            status = env.storage.status_now
            alt = -float(status[2])
            if alt < min_alt:
                min_alt = alt
            if save_traj:
                traj_states.append(np.copy(status))
                traj_actions.append(np.copy(action))
            if done:
                done_type = info.get('done_type', 'unknown')
                results[done_type] = results.get(done_type, 0) + 1
                results['rewards'].append(total_reward)
                results['steps'].append(step)
                altitude = -float(status[2])
                heading = float(status[8])
                flare = (float(status[22]) + float(status[24])) * 0.5
                wx, wy = float(status[28]), float(status[29])
                wind_dir = float(np.arctan2(wy, wx)) if (abs(wx) + abs(wy) > 1e-8) else 0.0
                target_heading = wind_dir + np.pi
                err = float(abs(angle_normalize(heading - target_heading, unit='rad')))
                results['final_altitudes'].append(altitude)
                results['heading_alignments'].append(float(np.cos(err)))
                results['final_flare'].append(flare)
                results['min_altitudes'].append(min_alt)
                all_details.append({
                    'worker_id': 0, 'ep': ep, 'done_type': done_type,
                    'reward': total_reward, 'steps': step,
                    'altitude': altitude, 'min_alt': min_alt,
                    'heading_err_deg': float(np.rad2deg(err)), 'flare': flare,
                })
                if save_traj:
                    all_trajectories.append({
                        'type': done_type,
                        'states': np.array(traj_states),
                        'actions': np.array(traj_actions),
                        'total_reward': total_reward,
                    })
                break

    env.close()
    saved = _print_results(results, num_episodes, seed, all_details,
                           all_trajectories, save_traj)
    if saved:
        save_path = model_path.replace('.pth', '_eval_trajectories.npz')
        np.savez_compressed(save_path, trajectories=saved)
        print(f'  Trajectories saved to: {save_path}')
    return results


def _evaluate_parallel(model_path, num_episodes, seed, save_traj, num_workers,
                       task_override=None):
    episodes_per_worker = [num_episodes // num_workers] * num_workers
    for i in range(num_episodes % num_workers):
        episodes_per_worker[i] += 1
    active_workers = sum(1 for e in episodes_per_worker if e > 0)
    print(f'  Parallel mode: {active_workers} workers, '
          f'{episodes_per_worker} episodes per worker')

    task_args = [
        (model_path, ep_count, seed, save_traj, wid, task_override)
        for wid, ep_count in enumerate(episodes_per_worker) if ep_count > 0
    ]
    t0 = time.time()
    with Pool(processes=active_workers) as pool:
        worker_outputs = pool.map(_worker_run_episodes, task_args)
    parallel_time = time.time() - t0

    merged, all_details, all_trajectories = _merge(worker_outputs, save_traj)
    saved = _print_results(merged, num_episodes, seed, all_details,
                           all_trajectories, save_traj)
    if saved:
        save_path = model_path.replace('.pth', '_eval_trajectories.npz')
        np.savez_compressed(save_path, trajectories=saved)
        print(f'  Trajectories saved to: {save_path}')
    print(f'\n  Parallel time: {parallel_time:.1f}s ({active_workers} workers)')
    return merged


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained HER+PPO model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model .pth file')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_traj', action='store_true')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default 1; use cpu_count//2 for speed)')
    parser.add_argument('--task', type=str, default=None,
                        help='task yml filename inside parafoil_env/config/task/ '
                             '(overrides framework config). Used for OOD evals, '
                             'e.g. --task flare_landing_ood_strong.yml')
    args = parser.parse_args()

    if not os.path.isabs(args.model):
        args.model = os.path.join(package_root, args.model)
    if not os.path.exists(args.model):
        print(f'Model not found: {args.model}')
        sys.exit(1)
    if args.render and args.workers > 1:
        print('Warning: --render is incompatible with --workers > 1; disabling render.')
        args.render = False

    nw = args.workers if args.workers > 0 else max(1, cpu_count() // 2)

    print(f'Loading model: {args.model}')
    if args.task:
        print(f'Task override : {args.task}')
    print(f'Running {args.episodes} evaluation episodes with {nw} worker(s)...\n')
    t0 = time.time()
    evaluate(args.model, args.episodes, args.render, args.seed, args.save_traj, nw,
             task_override=args.task)
    print(f'\nTotal time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
