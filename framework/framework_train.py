# -*- coding: utf-8 -*-
"""
Top-level training driver for HER+PPO parafoil flare landing.

The control flow is unchanged from the original framework:
  1) ``num_workers`` subprocesses roll out episodes in parallel, writing into
     their own slice of shared memory.
  2) The main process collects ``batch_size`` trajectories and signals the
     PPO agent process to perform an update.
  3) Logger process consumes the same trajectories for metrics + checkpoints.

What's new for HER:
  * Each worker now relabels its own episode in-place at ``done`` time (see
    ``worker/worker_ppo.py``) and tags relabeled episodes with done-code 4 so
    that the PPO update and the logger can distinguish HER from real success.
  * HER hyper-parameters are forwarded to each worker via ``env_args.her``
    populated below.
"""
import copy
import os
import sys
from types import SimpleNamespace

import numpy as np

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if package_root not in sys.path:
    sys.path.append(package_root)

from parafoil_env.utils.shared_memory_manager import SharedMemoryManager


class Training(object):
    smm_global: SharedMemoryManager
    smm_workers: list

    def __init__(self, cfg):
        self.task = None
        self.cfg = cfg
        self.episode_count = 0
        self.time_step_count = 0

        self._check_config()
        self._create_smm()
        self._create_processes()

    # ------------------------------------------------------------------
    def _check_config(self):
        if hasattr(self.cfg, 'proc'):
            proc = self.cfg.proc
            if proc.max_num_workers > proc.batch_size:
                proc.num_workers = proc.batch_size
                print(
                    f'<max_num_workers>:{proc.num_workers} larger than '
                    f'<batch_size>:{proc.batch_size}. Setting num_workers '
                    f'to {proc.batch_size}.\n',
                    end='',
                )
            else:
                proc.num_workers = proc.max_num_workers

    # ------------------------------------------------------------------
    def _create_smm(self):
        proc = self.cfg.proc
        from parafoil_env.utils.task import Task
        task = Task(proc.task_config)

        # col = obs s_t | action a_t | log_prob | next_obs s_{t+1} | reward | done
        col = proc.state_dim * 2 + proc.action_dim + 2 + 1

        worker_smm_items = [
            SimpleNamespace(name='args', dtype='variable', size=2048, value=SimpleNamespace()),
            SimpleNamespace(name='env_args', dtype='variable', size=1024 * 1024,
                            value=SimpleNamespace()),
            SimpleNamespace(name='data', dtype='ndarray_float32',
                            shape=(proc.max_episode_steps + 1, col)),
            SimpleNamespace(name='evaluation_index', dtype='ndarray_float32', shape=(20,)),
            SimpleNamespace(name='step', dtype='int', value=0),
            SimpleNamespace(name='lock_require_action', dtype='semaphore'),
            SimpleNamespace(name='lock_step', dtype='semaphore'),
            SimpleNamespace(name='lock_done', dtype='semaphore'),
            SimpleNamespace(name='lock_close', dtype='semaphore'),
        ]

        # HER settings live on cfg.her (set in framework_ppo.yml). We copy them
        # onto each worker's env_args so the worker can pick them up without a
        # second config load.
        her_cfg = getattr(self.cfg, 'her', None)

        self.smm_workers = []
        for i in range(proc.num_workers):
            self.smm_workers.append(
                SharedMemoryManager(create=True, items=copy.deepcopy(worker_smm_items))
            )
            self.smm_workers[i].wid = i
            worker_args = copy.deepcopy(self.cfg.worker)
            worker_args.id = i
            worker_args.render = True if i < proc.render_num else None
            self.smm_workers[i].args = worker_args
            env_args_i = task.acquire_env_args()
            if her_cfg is not None:
                env_args_i.her = her_cfg
            self.smm_workers[i].env_args = env_args_i

        global_smm_items = [
            SimpleNamespace(name='cfg', dtype='variable', size=4000, value=SimpleNamespace()),
            SimpleNamespace(name='policy', dtype='variable', size=2_000_000,
                            value=SimpleNamespace()),
            SimpleNamespace(name='trajectory_logs', dtype='ndarray_float32',
                            shape=(proc.batch_size, proc.max_episode_steps + 1, col)),
            SimpleNamespace(name='step_logs', dtype='ndarray_float32',
                            shape=(proc.batch_size,)),
            SimpleNamespace(name='evaluation_indexes', dtype='ndarray_float32',
                            shape=(proc.batch_size, 20)),
            SimpleNamespace(name='done', dtype='semaphore'),
            SimpleNamespace(name='lock_policy_ready', dtype='semaphore'),
            SimpleNamespace(name='lock_agent_step', dtype='semaphore'),
            SimpleNamespace(name='lock_logger_step', dtype='semaphore'),
            SimpleNamespace(name='lock_agent_close', dtype='semaphore'),
            SimpleNamespace(name='lock_logger_close', dtype='semaphore'),
        ]
        self.smm_global = SharedMemoryManager(create=True, items=global_smm_items)
        self.smm_global.cfg = self.cfg

    # ------------------------------------------------------------------
    def _create_processes(self):
        from multiprocessing import Process
        from worker.agent import PPOAgent
        from worker.logger import Logger
        from worker.worker_ppo import Worker

        proc = self.cfg.proc
        workers_items = [w.items for w in self.smm_workers]
        self.agent = Process(
            target=PPOAgent, args=(self.smm_global.items, workers_items), name='Agent'
        )
        self.logger = Process(target=Logger, args=(self.smm_global.items,), name='Logger')
        self.workers = [
            Process(target=Worker, args=(self.smm_workers[i].items,))
            for i in range(proc.num_workers)
        ]

    def _assigned_task(self, wid, start=False):
        worker = self.smm_workers[wid]
        env_args_i = self.task.acquire_env_args()
        her_cfg = getattr(self.cfg, 'her', None)
        if her_cfg is not None:
            env_args_i.her = her_cfg
        worker.env_args = env_args_i
        worker.step = 0
        if start:
            worker.lock_step.release()

    def _collect_data(self):
        proc = self.cfg.proc
        [self._assigned_task(wid=wid, start=False) for wid in range(proc.num_workers)]
        [worker.lock_step.release() for worker in self.smm_workers]
        started_count = proc.num_workers
        log_count = 0

        while True:
            for wid, worker in enumerate(self.smm_workers):
                if worker.lock_done.acquire(block=False):
                    step = worker.step
                    self.smm_global.trajectory_logs[log_count, :step, :] = worker.data[:step, :]
                    self.smm_global.step_logs[log_count] = step
                    self.smm_global.evaluation_indexes[log_count, :] = worker.evaluation_index
                    log_count += 1

                    if started_count < proc.batch_size:
                        self._assigned_task(wid=wid, start=True)
                        started_count += 1

                if log_count == proc.batch_size:
                    return True
            self._sub_process_alive_check()

    def run(self):
        proc = self.cfg.proc
        from parafoil_env.utils.task import Task
        self.task = Task(proc.task_config)

        self.logger.start()
        self.agent.start()
        [w.start() for w in self.workers]
        while True:
            assert self._collect_data() is True
            self.episode_count += int(proc.batch_size)
            self.smm_global.lock_agent_step.release()
            self.smm_global.lock_logger_step.release()

            if self.episode_count > proc.max_episode or self.time_step_count > proc.max_time_step:
                print(
                    f'total step: {self.time_step_count}, total episode: {self.episode_count}\n',
                    end='',
                )
                self._close()

    def _sub_process_alive_check(self):
        terminate = False
        if not self.agent.is_alive():
            print('Agent dead.', flush=True); terminate = True
        if not self.logger.is_alive():
            print('Logger dead.', flush=True); terminate = True
        for wid, worker in enumerate(self.workers):
            if not worker.is_alive():
                print(f'Worker_{wid} dead.', flush=True); terminate = True
        if terminate:
            self._close()

    def _terminate(self):
        try:
            self.agent.kill()
            [p.kill() for p in self.workers]
            self.logger.kill()
            print('process killed.', flush=True)
        except Exception as e:
            print(e, flush=True)
        try:
            self.agent.join()
            [p.join() for p in self.workers]
            self.logger.join()
        except Exception as e:
            print(e, flush=True)
        exit(0)

    def _close(self):
        self.smm_global.lock_agent_close.release()
        self.smm_global.lock_logger_close.release()
        for worker in self.smm_workers:
            worker.lock_close.release()
            worker.lock_step.release()
        self.agent.join()
        self.logger.join()
        [w.join() for w in self.workers]
        exit(0)


def main():
    import multiprocessing as mp
    mp.set_start_method('spawn')

    # Quiet some noisy backends on headless servers.
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

    from parafoil_env.utils.config import Config

    # Allow overriding the framework config from CLI / env-var so that the
    # tmux entry script (tools/start_train.sh) can point at debug vs prod yml
    # without editing source.
    framework_config_path = os.environ.get(
        'FRAMEWORK_CONFIG_PATH',
        f'{package_root}/config/framework_ppo.yml',
    )
    if len(sys.argv) > 1 and sys.argv[1] not in ('-h', '--help'):
        framework_config_path = sys.argv[1]

    framework_config = Config(framework_config_path).get_args()
    if hasattr(framework_config, 'proc'):
        task_config_path = (
            f'{package_root}/parafoil_env/config/task/{framework_config.proc.task_path}'
        )
        task_config = Config(task_config_path).get_args()
        from parafoil_env.utils.task import Task
        env_args = Task(task_config).acquire_env_args()
        framework_config.proc.state_dim = env_args.observation.dim
        framework_config.proc.action_dim = env_args.action.dim
        framework_config.proc.max_episode_steps = env_args.max_episode_steps
        framework_config.proc.evaluation_index = env_args.evaluation_index
        framework_config.proc.task_config = task_config
    else:
        raise NotImplementedError
    if hasattr(framework_config, 'log'):
        framework_config.log.framework_config_path = framework_config_path
        framework_config.log.task_config_path = task_config_path

    training = Training(framework_config)
    try:
        training.run()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: shutting down workers...', flush=True)
        try:
            training._close()
        except Exception as e:
            print(f'Graceful close failed: {e}; forcing terminate.', flush=True)
            try:
                training._terminate()
            except Exception:
                pass
    except Exception as e:
        print(f'Unhandled exception: {e}', flush=True)
        try:
            training._terminate()
        except Exception:
            pass


if __name__ == '__main__':
    main()
