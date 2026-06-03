# -*- coding: utf-8 -*-
"""
Created on 2022/6/8 11:17

@author: qk
"""
import os
import pandas as pd
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Semaphore
from types import SimpleNamespace

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from worker.worker_ppo import Worker


class Controller:
    def __init__(self, coordinates, closest_targets, targets):
        from algorithms.pid import PID
        self.last_last_coordinates = coordinates[0]
        self.last_coordinates = coordinates[1]
        self.last_last_targets = targets[0].reshape(3, 3)
        self.last_targets = targets[1].reshape(3, 3)
        self.last_last_closest_target = closest_targets[0]
        self.last_closest_target = closest_targets[1]

        # self.pid = PID2(kp=0.42, ki=0.002, kd=0.172)
        last_error = self._calculate_error(self.last_coordinates, self.last_targets, self.last_last_coordinates)
        self.pid = PID(kp=0.5, ki=0.0005, kd=0.5, init_last_error=last_error, interval=1., limit=0.6)

    def step(self, coordinates, closest_target, targets):
        """

        :param coordinates: 当前坐标
        :param closest_target: 最近目标点坐标
        :param targets: z,z+10,z+20目标点坐标
        :return:
        """
        error = self._calculate_error(coordinates, targets, self.last_coordinates)
        output = self.pid.step(error)

        deflection = [0, output] if output > 0 else [-output, 0]
        self._update(coordinates, closest_target, targets)
        return deflection

    def _update(self, coordinates, closest_target, targets):
        self.last_last_coordinates = self.last_coordinates
        self.last_coordinates = coordinates
        self.last_last_targets = self.last_targets
        self.last_targets = targets
        self.last_last_closest_target = self.last_closest_target
        self.last_closest_target = closest_target

    @staticmethod
    def _calculate_error(coordinates, targets, last_coordinates):
        from parafoil_env_basic.utils.method import vector2radian, angle_normalize
        target, target_10, target_20 = targets

        rad1 = vector2radian(target[:2] - last_coordinates[:2])
        rad2 = vector2radian(coordinates[:2] - last_coordinates[:2])

        err = angle_normalize(rad1 - rad2, unit='rad')
        return err


class WorkerPID(Worker):
    def __init__(self, worker_args):
        self.agent = None
        if not os.path.exists(package_root + '/data'):
            os.mkdir(package_root + r'/data')
        if not os.path.exists(package_root + '/data/evaluate_pid'):
            os.mkdir(package_root + r'/data/evaluate_pid')

        super(WorkerPID, self).__init__(worker_args)

    def reset(self):
        """"""
        e_args = self.env_args
        self._acquire_signal()
        tj_num = int(self.cmd[5])
        if 'mode' in e_args.trajectory_info and e_args.trajectory_info['mode'] == 'assigned':
            e_args.trajectory_info['tj_id'] = tj_num
            preset_name = None
        else:
            preset_name = f'{self.args.gym_name}/{str(tj_num).zfill(4)}'
            e_args.trajectory_info['preset_name'] = preset_name
        # reset agent
        states = self.env.reset(self.env_args)
        self.agent = Controller(states[:2, :3], states[:2, 3:6], states[:2, 6:15])
        # other
        self._write_state(state=states[0], step=-1)
        self.step[:] = 0
        self._set_ready_flag()  # require action
        # print(f'worker {self.id} started: {tj_num}')
        return states[0], tj_num, preset_name

    def run(self):
        e_args = self.env_args
        if e_args.render:
            self.env.render()

        while True:
            """ episode """
            """ reset：重新选择轨迹 """
            obs, tj_num, preset_name = self.reset()
            while True:
                """ step """
                action = self.agent.step(obs[:3], obs[3:6], obs[6:15].reshape(3, 3))
                step = int(self.step[0])
                obs, reward, done, info = self.env.step(action)
                self._write_state(obs, step)
                self._write_reward_done(reward=reward, done=done, step=step)
                self.step[:] += 1

                if self._is_done(step=step, info=info):
                    assert self.data[step, -1] != 0
                    self._set_done_flag()
                    if self.args.env_args.mode == 'evaluate':
                        from parafoil_env_basic.utils.data_io import write_file
                        if 'mode' in e_args.trajectory_info and e_args.trajectory_info['mode'] == 'assigned':
                            trajectory = SimpleNamespace(trajectory=self.env.storage.status_log_now, env_args=e_args)
                            file_path = f'{self.args.storage_path}/{str(tj_num).zfill(4)}.dat.xz'
                        else:
                            trajectory = self.env.storage.status_log_now
                            file_path = f'{self.args.storage_path}/{preset_name}.dat.xz'
                        write_file(file_name=file_path, data=trajectory, compress=True)
                        print(f'worker {self.id} save file at {file_path}')

                    # print(f'worker {self.id}, index:{int(self.cmd[4])} done at step: {step}')
                    break
                else:
                    assert self.step[:] < e_args.max_episode_steps
                    self._set_ready_flag()  # require action
            self.step_lock.release()
            # exit(0)


def main():
    from config.basic_parameter import Config
    from parafoil_env_basic.utils.config import Config as EnvConfig
    framework_config_path = package_root + '/config/framework_pid.yml'
    cfg = Config(yml_path=framework_config_path).get_args()
    env_config_path = package_root + r'/parafoil_env_basic/data/config/' + cfg.env_cfg_file
    list_env_cfg = EnvConfig(path=env_config_path).get_args()

    np.random.seed(cfg.random_seed)
    for env_args in list_env_cfg:
        # set random_seed and max_episode_steps
        env_args.random_seed = np.random.randint(200)
        env_args.max_episode_steps = cfg.max_episode_steps

    cfg.framework_config_path = framework_config_path
    cfg.env_config_path = env_config_path

    cfg.obs_dim = list_env_cfg[0].obs_dim
    cfg.action_dim = list_env_cfg[0].action_dim
    cfg.batch_size = cfg.evaluate_episode

    import warnings
    if cfg.max_num_workers != len(list_env_cfg):
        warnings.warn(
            f"max_num_workers:{cfg.max_num_workers} does not agree with list_env_args:{len(list_env_cfg)}.")
    cfg.num_workers = min(cfg.max_num_workers, len(list_env_cfg))
    if cfg.num_workers > cfg.batch_size:
        warnings.warn(f'<num_workers>:{cfg.num_workers} larger than <batch_size>:{cfg.batch_size}.\n'
                      f'reset <num_workers> to: {cfg.batch_size}')
        cfg.num_workers = cfg.batch_size

    col = cfg.obs_dim * 2 + cfg.action_dim + 2 + 1
    cfg.shm_info = [{'name': 'cmd', 'shape': (cfg.num_workers, 10, 1), 'dtype': np.float32},
                    {'name': 'data', 'shape': (cfg.num_workers, cfg.max_episode_steps + 1, col), 'dtype': np.float32}]
    shm_list = []
    for index, info in enumerate(cfg.shm_info):
        try:
            shm_size = 1 * 4
            for i in info['shape']:
                shm_size *= i
            shm_size = int(np.ceil(shm_size / 4096) * 4096)
        except Exception as e:
            shm_size = info['size']
        shm_list.append(SharedMemory(create=True, size=shm_size))
        cfg.shm_info[index]['shm_name'] = shm_list[index].name

    from types import SimpleNamespace
    worker_args = SimpleNamespace()
    worker_args.worker_id = 0
    worker_args.shm_info = cfg.shm_info
    list_env_cfg[0].render = True
    worker_args.env_args = list_env_cfg[0]
    worker_args.locks = [Semaphore(0) for i in range(2)]
    worker_args.mode = 'evaluate'
    worker_args.gym_name = cfg.gym_name
    worker_args.args = cfg

    worker_args.locks[1].release()
    w = WorkerPID(worker_args)

    exit(0)


if __name__ == '__main__':
    main()
