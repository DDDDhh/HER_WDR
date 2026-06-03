# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 15:25

@author: qk
"""
import copy

import numpy as np
from multiprocessing.shared_memory import SharedMemory
import pickle
import os
import subprocess
from types import SimpleNamespace
from algorithms.ppo import PPO
# from algorithms.ppo_v44 import PPO_v44
import os
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from parafoil_env.utils.shared_memory_manager import SharedMemoryManager


class PPOAgent:
    def __init__(self, global_items, workers_items):
        print(f'Agent pid: {os.getpid()}')
        self.smm = SharedMemoryManager(items=global_items)
        self.smm_workers = [SharedMemoryManager(items=worker_items) for worker_items in workers_items]
        self.cfg = self.smm.cfg
        self.agent = PPO(self.cfg)

        proc = self.cfg.proc
        if hasattr(proc, 'model_path') and proc.model_path is not None:
            self.agent.load(f'{package_root}/{proc.model_path}')
        self._write_policy()

        try:
            self.run()
        except subprocess.CalledProcessError as e:
            print(f'******************************************************\nagent Error, {e.output}\n', end='')
            raise ValueError(f'Agent: Something went wrong...')
        finally:
            print('Agent finally.\n', end='')

    def _update_policy(self):
        agent = self.cfg.agent
        self._write_policy()
        experiences = self._process_data()
        self.agent.update(experiences)
        self.agent.decay_action_std(agent.action_std_decay_rate, agent.min_action_std)
        self.agent.scheduler.step()

    def _write_policy(self):
        self.smm.policy = copy.deepcopy(self.agent)
        self.smm.lock_policy_ready.release()

    def _process_data(self):
        data_lens = self.smm.step_logs
        num_data_row = self.smm.trajectory_logs.shape[-1]
        experiences = np.zeros((data_lens.sum(dtype=int), num_data_row), dtype=np.float32)
        index = 0
        for i, log in enumerate(self.smm.trajectory_logs):
            experiences[index:index + int(data_lens[i]), :] = self.smm.trajectory_logs[i, :int(data_lens[i]), :]
            index += int(data_lens[i])
        return experiences

    def run(self):
        proc = self.cfg.proc
        sampling = False if proc.mode == 'evaluate' else True
        while True:
            for wid, worker in enumerate(self.smm_workers):
                if worker.lock_require_action.acquire(block=False):
                    step = worker.step
                    state = worker.data[step, :proc.state_dim]



                    # 上层策略选择底层策列根据state
                    # if option_termination:
                    # option = policy_over_options.sample(state)


                    action, action_logprob = self.agent.select_action(state, sampling=sampling)
                    # 预测是否终止

                    worker.data[step, proc.state_dim:proc.state_dim+proc.action_dim+1] = np.append(action, action_logprob)

                    # option_termination = self.agent.predict_option_termination(state)
                    # if option_termination:
                        # 更新
                    # self._update_policy()
                    worker.lock_step.release()
                    # if step > 2 :
                        # self.smm.lock_agent_step.release()
            if self.smm.lock_agent_step.acquire(block=False):
                self._update_policy()
            if self.smm.lock_agent_close.acquire(block=False):
                self.close()

    def close(self):
        print('Edited: Agent.')
        exit(0)

