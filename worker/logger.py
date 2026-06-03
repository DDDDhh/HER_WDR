# -*- coding: utf-8 -*-
"""
Logger process: consumes the same batch of trajectories the agent updates on,
writes metrics to TensorBoard, and persists checkpoints.

HER changes
-----------
* Counts done-codes separately: dead(1), timeout(2), real complete(3),
  HER-relabeled(4). The HER count is logged but NOT used to decide which
  checkpoint is "best" — otherwise the metric collapses to "are we relabeling
  a lot?", which says nothing about the real policy quality.
"""
import copy
import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_root) if package_root not in sys.path else None

from parafoil_env.utils.shared_memory_manager import SharedMemoryManager
from parafoil_env.utils.data_io import TarWrite


# Mirror of the done codes in worker_ppo.py / algorithms/ppo.py
DONE_FLYING = 0
DONE_DEAD = 1
DONE_TIMEOUT = 2
DONE_COMPLETE = 3
DONE_HER = 4


class Logger(object):
    def __init__(self, smm_items):
        self.smm = SharedMemoryManager(items=smm_items)
        print(f'Logger pid: {os.getpid()}', flush=True)
        self.cfg = self.smm.cfg
        proc = self.cfg.proc
        log = self.cfg.log

        for sub in ('data', 'data/logs', 'data/runs'):
            path = os.path.join(package_root, sub)
            os.makedirs(path, exist_ok=True)

        if log.log_name is None:
            log.log_name = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            os.makedirs(os.path.join(package_root, 'data', 'logs', log.log_name), exist_ok=True)
            os.makedirs(os.path.join(package_root, 'data', 'logs', log.log_name, 'model'),
                        exist_ok=True)
            self.trajectory_tar = TarWrite(
                f'{package_root}/data/logs/{log.log_name}/trajectory.xz.tar', None, 'xz'
            )

        log.sw_path = os.path.join(package_root, 'data', 'runs', log.log_name)
        log.log_path = os.path.join(package_root, 'data', 'logs', log.log_name)
        self.sw = SummaryWriter(log.sw_path)

        # Copy configs for reproducibility.
        import shutil
        shutil.copyfile(
            log.framework_config_path,
            os.path.join(log.log_path, os.path.basename(log.framework_config_path)),
        )
        shutil.copyfile(
            log.task_config_path,
            os.path.join(log.log_path, os.path.basename(log.task_config_path)),
        )

        self.max_ep_avg_reward = -1e10
        self.max_success_rate = -1.0
        self.episode_count = 0
        self.step_count = 0

        self.log_labels = [' '] * (proc.action_dim + proc.state_dim * 2 + 3)
        self.log_labels[0] = 'obs'
        self.log_labels[proc.state_dim] = 'action'
        self.log_labels[proc.state_dim + 2] = 'action_prob'
        self.log_labels[proc.state_dim + 3] = 'n_obs'
        self.log_labels[-2] = 'reward'
        self.log_labels[-1] = 'done'

        self.agent = self._update_model()
        try:
            self.run()
        except Exception:
            import traceback
            print(f'logger error:\n{traceback.format_exc()}', flush=True)
            exit()

    # ------------------------------------------------------------------
    def _update_model(self):
        self.smm.lock_policy_ready.acquire()
        return self.smm.policy

    def _process_data(self):
        data_lens = self.smm.step_logs
        num_data_row = self.smm.trajectory_logs.shape[-1]
        experiences = np.zeros((int(data_lens.sum()), num_data_row), dtype=np.float32)
        index = 0
        for i in range(len(data_lens)):
            n = int(data_lens[i])
            trajectory = self.smm.trajectory_logs[i, :n, :]
            experiences[index:index + n, :] = trajectory
            self.trajectory_tar.add(f'{str(self.episode_count)}.xz', trajectory)
            self.episode_count += 1
            index += n
        self.step_count += (int(data_lens.sum()) - self.cfg.proc.batch_size)
        return experiences

    # ------------------------------------------------------------------
    def run(self):
        proc = self.cfg.proc
        while True:
            self._wait_signal()
            self.agent = self._update_model()
            experiences = self._process_data()

            std = np.sqrt(self.agent.policy.action_var.cpu().numpy())
            lr_critic = float(self.agent.optimizer.state_dict()['param_groups'][0]['lr'])
            lr_actor = float(self.agent.optimizer.state_dict()['param_groups'][1]['lr'])

            sum_rewards = experiences[:, -2].sum()
            episode_avg_reward = sum_rewards / proc.batch_size

            done_codes = experiences[:, -1].astype(np.int32).tolist()
            dead_n = done_codes.count(DONE_DEAD)
            timeout_n = done_codes.count(DONE_TIMEOUT)
            success_n = done_codes.count(DONE_COMPLETE)
            her_n = done_codes.count(DONE_HER)
            total_done = dead_n + timeout_n + success_n + her_n

            step_logs = self.smm.step_logs
            ep_max_step = float(np.max(step_logs))
            ep_avg_step = float(np.mean(step_logs))

            print(
                f'Episode: {self.episode_count}  step: {self.step_count}  '
                f'AvgR: {episode_avg_reward:7.3f}  std: {std[0]:.3f}  '
                f'done {total_done}/{proc.batch_size} '
                f'(dead {dead_n}, timeout {timeout_n}, success {success_n}, her {her_n})  '
                f'avg_step: {ep_avg_step:.1f}',
                flush=True,
            )

            self.sw.add_scalar('RL/Episode Average Reward', episode_avg_reward,
                               self.episode_count)
            self.sw.add_scalar('RL/std', float(std[0]), self.episode_count)
            self.sw.add_scalar('RL/avg step', ep_avg_step, self.episode_count)
            self.sw.add_scalar('RL/max step', ep_max_step, self.episode_count)
            self.sw.add_scalar('RL/done episode', total_done, self.episode_count)
            self.sw.add_scalar('RL/success episode', success_n, self.episode_count)
            self.sw.add_scalar('RL/her episode', her_n, self.episode_count)
            self.sw.add_scalar('RL/dead episode', dead_n, self.episode_count)
            self.sw.add_scalar('RL/timeout episode', timeout_n, self.episode_count)
            self.sw.add_scalar('RL/real success rate',
                               success_n / max(1, proc.batch_size), self.episode_count)
            self.sw.add_scalar('RL/her rate', her_n / max(1, proc.batch_size),
                               self.episode_count)
            self.sw.add_scalar('RL/lr_actor', lr_actor, self.episode_count)
            self.sw.add_scalar('RL/lr_critic', lr_critic, self.episode_count)
            self._evaluate_index()

            self.save_model(episode_avg_reward, success_n, proc.batch_size)

    def _evaluate_index(self):
        if hasattr(self.cfg.proc, 'evaluation_index'):
            evaluation_indexes = self.smm.evaluation_indexes
            for i, idx_name in enumerate(self.cfg.proc.evaluation_index):
                self.sw.add_scalar(f'RL/{idx_name}', float(evaluation_indexes[:, i].mean()),
                                   self.episode_count)

    # ------------------------------------------------------------------
    def save_model(self, episode_avg_reward, success_count, batch_size):
        proc = self.cfg.proc
        log = self.cfg.log
        success_rate = success_count / max(1, batch_size)

        # Best = highest REAL success rate; ties broken by avg reward. HER is
        # excluded on purpose so the checkpoint reflects actual policy quality.
        is_best = (
            success_rate > self.max_success_rate
            or (success_rate == self.max_success_rate and episode_avg_reward > self.max_ep_avg_reward)
        )
        if is_best:
            self.max_ep_avg_reward = episode_avg_reward
            self.max_success_rate = success_rate
            best_model_path = os.path.join(log.log_path, 'model', 'best_model.pth')
            checkpoint = {
                'Episode': self.episode_count,
                'Time_step': self.step_count,
                'Policy': self.agent.policy_old.state_dict(),
                'action_std': self.agent.policy_old.action_var.cpu().numpy()[0],
            }
            torch.save(checkpoint, best_model_path)
            print(
                f'---- best model (success={success_rate:.1%}, avgR={episode_avg_reward:.2f}) '
                f'saved at {best_model_path}',
                flush=True,
            )

        if self.episode_count % (proc.batch_size * log.model_save_interval) == 0:
            model_path = os.path.join(
                log.log_path, 'model',
                f'{time.strftime("%Y-%m-%d_%H-%M", time.localtime())}.pth',
            )
            checkpoint = {
                'Episode': self.episode_count,
                'Time_step': self.step_count,
                'Policy': self.agent.policy_old.state_dict(),
                'action_std': self.agent.policy_old.action_var.cpu().numpy()[0],
            }
            torch.save(checkpoint, model_path)
            print(f'---- model snapshot saved at: {model_path}', flush=True)

    def _wait_signal(self):
        self.smm.lock_logger_step.acquire()
        if self.smm.lock_logger_close.acquire(block=False):
            self.close()

    def close(self):
        self.trajectory_tar.close()
        print('Exited: Logger.', flush=True)
        exit(0)


def main():
    pass


if __name__ == '__main__':
    main()
