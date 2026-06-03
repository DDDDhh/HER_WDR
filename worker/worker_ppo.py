# -*- coding: utf-8 -*-
"""
Worker process: runs one parafoil episode at a time and writes the resulting
trajectory to shared memory. When HER is enabled and the episode did not end
in a real `complete`, the worker relabels the trajectory in-place under a
hindsight wind direction and tags it with ``done_code = 4`` so that:

* the agent (PPO) still treats it as terminal,
* the logger can count HER episodes separately from real successes.

Design notes vs. the previous in-place HER (folder 1):
  * Uses **pure functions** from ``parafoil_env.utils.her`` / ``EnvFlare``
    instead of mutating ``env.args`` and ``env.storage`` through a Proxy.
  * Records the per-step phys state ONCE during the episode (no expensive
    re-rolling); HER relabel is O(K) extra math at the end of an episode.
  * Wind direction is read from the stored wind vector ``status[28:30]`` —
    so HER works regardless of whether the task's wind config is in degrees
    (dataset) or radians.
"""
import os
import sys

import numpy as np

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if package_root not in sys.path:
    sys.path.append(package_root)

from parafoil_env.utils.shared_memory_manager import SharedMemoryManager
from parafoil_env.utils import her as her_lib


# done codes — keep in sync with logger.py / algorithms/ppo.py
DONE_FLYING = 0.0
DONE_DEAD = 1.0
DONE_TIMEOUT = 2.0
DONE_COMPLETE = 3.0
DONE_HER = 4.0


class Worker:
    def __init__(self, items):
        self.smm = SharedMemoryManager(items=items)
        self.args = self.smm.args
        print(f'Worker_{self.args.id} pid: {os.getpid()}', flush=True)

        env_args = self.smm.env_args
        Env = env_args.env_class
        self.env = Env(env_args)

        # Per-worker RNG — uses worker id to give each worker a unique HER
        # decision stream while remaining deterministic for a given run.
        self._rs = np.random.RandomState(seed=int(self.args.id) * 7919 + 11)

        try:
            self.run()
        except Exception as e:
            print(f'Worker {self.args.id} error: {e}', flush=True)
            import traceback
            traceback.print_exc()
            raise
        finally:
            print(f'Worker {self.args.id} exited.', flush=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        args = self.args
        self.env.render() if args.render else None

        while True:
            self._acquire_signal()
            env_args = self.smm.env_args
            init_state = self.env.reset(env_args)
            self._write_state(
                step=-1, state=init_state,
                obs_dim=env_args.observation.dim,
                action_dim=env_args.action.dim,
            )
            self.smm.step = 0
            self._set_ready_flag()

            while True:
                step = self.smm.step
                self._acquire_signal()
                action = self._read_action(
                    step, env_args.observation.dim, env_args.action.dim
                )
                state, reward, done, info = self.env.step(action)

                self._write_state(
                    step, state,
                    env_args.observation.dim, env_args.action.dim,
                )
                self._write_reward_done(step=step, reward=reward, done=done)
                self.smm.step = step + 1

                if done:
                    done_type = info.get('done_type', 'dead')
                    if done_type == 'complete':
                        self._write_done(step, DONE_COMPLETE)
                    elif done_type == 'done':
                        self._write_done(step, DONE_TIMEOUT)
                    else:
                        self._write_done(step, DONE_DEAD)

                    self._maybe_apply_her(
                        done_type=done_type,
                        last_step_index=step,
                        env_args=env_args,
                    )

                    eval_vals = [info.get(idx, 0) for idx in env_args.evaluation_index]
                    self.smm.evaluation_index[:len(eval_vals)] = eval_vals
                    self.smm.lock_done.release()
                    break
                else:
                    assert self.smm.step < env_args.max_episode_steps
                    self._set_ready_flag()

    # ------------------------------------------------------------------
    # HER
    # ------------------------------------------------------------------
    def _maybe_apply_her(self, done_type, last_step_index, env_args):
        her_cfg = getattr(env_args, 'her', None)
        # Allow disabling HER from outside (config, env override). Default ON.
        enable_her = True
        trigger_prob = 0.8
        min_steps = 5
        min_alt_descended = 20.0
        noise_std = 0.0
        if her_cfg is not None:
            enable_her = bool(getattr(her_cfg, 'enable', enable_her))
            trigger_prob = float(getattr(her_cfg, 'trigger_prob', trigger_prob))
            min_steps = int(getattr(her_cfg, 'min_steps', min_steps))
            min_alt_descended = float(getattr(her_cfg, 'min_alt_descended', min_alt_descended))
            noise_std = float(getattr(her_cfg, 'noise_std', noise_std))
        if not enable_her:
            return

        K = last_step_index + 1  # number of RL steps executed
        rate = int(env_args.action.rate)
        max_episode_steps = int(env_args.max_episode_steps)

        # Collect (K+1) phys states at RL-step boundaries: indices 0, rate, ..., K*rate.
        status_logs = self.env.storage.status_logs
        phys_states = []
        for k in range(K + 1):
            phys_states.append(np.copy(status_logs[k * rate]))

        if not her_lib.should_apply_her(
            phys_states=phys_states,
            done_type=done_type,
            min_steps=min_steps,
            min_alt_descended=min_alt_descended,
            trigger_prob=trigger_prob,
            rs=self._rs,
        ):
            return

        # Compute relabeled obs + rewards under the hindsight wind direction.
        # Any exception here is local to this episode — fall back to the
        # original labels rather than killing the worker.
        try:
            relabel = her_lib.relabel_trajectory(
                env_pure=self.env.__class__,   # static methods live on the class
                phys_states=phys_states,
                action_rate=rate,
                max_episode_steps=max_episode_steps,
                reward_kwargs=getattr(self.env, '_reward_kwargs', None),
                noise_std=noise_std,
                rs=self._rs,
            )
        except Exception as exc:
            print(f'[HER ERR] worker {self.args.id}: relabel failed ({exc!r}); '
                  f'keeping original labels.', flush=True)
            return

        new_obs = relabel['obs']          # (K+1, obs_dim)
        new_rew = relabel['rewards']      # (K,)
        obs_dim = int(env_args.observation.dim)
        act_dim = int(env_args.action.dim)

        # Sanity: shapes
        if new_obs.shape != (K + 1, obs_dim):
            print(
                f'[HER WARN] worker {self.args.id} obs shape mismatch '
                f'{new_obs.shape} != ({K+1},{obs_dim}); skipping',
                flush=True,
            )
            return

        # Overwrite obs / next_obs / reward / done in shared memory in place.
        # Action and logprob columns are intentionally left untouched — they
        # were sampled under the *original* observation but PPO's clipping +
        # the small obs delta from a wind rotation keeps the off-policy bias
        # bounded in practice.
        for k in range(K):
            self.smm.data[k, :obs_dim] = new_obs[k]
            self.smm.data[
                k,
                obs_dim + act_dim + 1: obs_dim * 2 + act_dim + 1
            ] = new_obs[k + 1]
            self.smm.data[k, -2] = float(new_rew[k])
            # Intermediate steps: keep continuing flag (0). Only the last step
            # gets the HER terminal marker.
            if k < last_step_index:
                self.smm.data[k, -1] = DONE_FLYING

        self.smm.data[last_step_index, -1] = DONE_HER

    # ------------------------------------------------------------------
    # Shared memory helpers (unchanged shape & semantics)
    # ------------------------------------------------------------------
    def _acquire_signal(self):
        self.smm.lock_step.acquire()
        if self.smm.lock_close.acquire(block=False):
            self.close()

    def _is_done(self, step):
        return bool(self.smm.data[step, -1])

    def _write_state(self, step, state, obs_dim, action_dim):
        if step != -1:
            self.smm.data[
                step,
                obs_dim + action_dim + 1: obs_dim * 2 + action_dim + 1
            ] = state
        self.smm.data[step + 1, :obs_dim] = state

    def _write_reward_done(self, step, reward, done):
        self.smm.data[step, -2] = float(reward)
        self.smm.data[step, -1] = DONE_FLYING if not done else 1.0  # overridden below

    def _write_done(self, step, done_code):
        self.smm.data[step, -1] = float(done_code)

    def _read_action(self, step, obs_dim, action_dim):
        return self.smm.data[step, obs_dim:obs_dim + action_dim]

    def _set_ready_flag(self):
        self.smm.lock_require_action.release()

    def close(self):
        self.env.close()
        print(f'Exited: Worker_{self.args.id}.', flush=True)
        exit(0)


class WorkerConfig(object):
    def __init__(self, global_args):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--id', type=int, default=0)
        parser.add_argument('--shm_info', default=0)
        self.env_args = parser.parse_args()
        self.semaphore = []


if __name__ == '__main__':
    pass
