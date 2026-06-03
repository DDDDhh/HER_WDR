# -*- coding: utf-8 -*-
"""
Created on 2022/6/23 15:28

@author: qk
"""

import sys
import os
import types

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_root)

from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process

import numpy as np
import pandas as pd
import subprocess

from parafoil_env_basic.trajectory_basic_modules.trajectory import Trajectory


class Analyzer(object):
    def __init__(self, args):
        self.args = args
        self.id = args.id
        print(args.id, args.preset_name)

        shm_info = args.shm_info
        self.shm = SharedMemory(name=shm_info['name'])
        self.data_array = np.ndarray(shape=shm_info['shape'], dtype=shm_info['data_type'], buffer=self.shm.buf)
        data = self.data_array[self.id, :]
        try:
            results = self.analyze(args.file_path, args.preset_name)
        except subprocess.CalledProcessError as e:
            print(f'******************************************************\nworker {self.id} Error, {e.output}')
            raise ValueError(f'worker {self.id}: Something went wrong...')
        data[0] = results[:, 1].mean()
        data[1] = results[:, 2].mean()
        data[2] = results[-1, 3]

        # self.shm.unlink()
        args.lock.release()
        print(data)

    def analyze(self, file_path, preset_name):
        from tqdm import tqdm
        parafoil_trajectory = pd.read_csv(file_path)     # 实际飞行轨迹
        parafoil_coordinates = parafoil_trajectory[parafoil_trajectory.columns[1:4]].to_numpy()     # 实际飞行坐标序列

        trajectory_info = dict(start_altitude=-8500, end_altitude=-50, interval=1, preset_name=preset_name)
        target_trajectory = Trajectory(trajectory_info)     # 目标轨迹
        target_coordinates = target_trajectory.get_trajectory()     # 目标坐标序列(裁剪后的）

        results = np.zeros((len(parafoil_coordinates), 4))
        results[:, 0] = parafoil_coordinates[:, 2]

        for i, coordinate in enumerate(parafoil_coordinates):
            target_coordinate = self._get_target_coordinate(coordinate, target_trajectory)
            results[i, 1] = np.linalg.norm(coordinate - target_coordinate)

            closest_target = self.find_closest_target(target_trajectory, coordinate)
            results[i, 2] = np.linalg.norm(coordinate - closest_target)

        return results

    @staticmethod
    def _get_target_coordinate(coordinate, target_trajectory):
        target_coordinate = np.zeros(3)
        target_coordinate[2] = coordinate[2]
        target_coordinate[:2] = target_trajectory.get_target_point(coordinate[2])
        return target_coordinate

    @staticmethod
    def find_closest_target(target_trajectory, coordinate):
        target_coordinates = target_trajectory.get_trajectory()  # 目标坐标序列
        # 找到相同z坐标的目标点 target_coordinate
        current_z = coordinate[2]
        target_coordinate = np.zeros(3)
        target_coordinate[2] = current_z
        target_coordinate[:2] = target_trajectory.get_target_point(current_z)
        # 计算到target_coordinate的距离的两倍作为区间宽度，在区间宽度中找最近点
        dist = np.linalg.norm(coordinate - target_coordinate)
        max_index = np.argmin(abs(target_coordinates[:, 2] - (current_z + dist)))
        min_index = np.argmin(abs(target_coordinates[:, 2] - (current_z - dist)))
        rg = target_coordinates[min_index:max_index + 1]
        dists = np.linalg.norm(rg - coordinate, axis=1)
        index = np.argmin(dists) + min_index
        # 找到最近点索引 index
        # 在宽度为2米的区间上，找更精确地最近点，每米100细分（diff）
        diff = 100
        max_z = target_coordinates[index - 1, 2]
        min_z = target_coordinates[index + 1, 2]
        zs = np.linspace(max_z, min_z, int((max_z - min_z) * -diff) + 1)
        target_coordinates = np.zeros((int((max_z - min_z) * -diff) + 1, 3))
        target_coordinates[:, 2] = zs
        target_coordinates[:, 0], target_coordinates[:, 1] = target_trajectory.get_target_point(zs)
        dists = np.linalg.norm(target_coordinates - coordinate, axis=1)
        index = np.argmin(dists)

        return target_coordinates[index]


def _get_preset_name(file_path):
    index = len(file_path)
    for j in range(2):
        index = file_path[:index].rfind('/')
    end_index = file_path.find('.csv')
    return file_path[index + 1:end_index]


def main():
    from functools import reduce
    args = types.SimpleNamespace()
    args.id = 0
    shm_info = dict(name='data', shape=(1, 3), data_type=np.float32)
    shm_size = reduce(lambda x, y: x * y, shm_info['shape']) * 4
    shm_size = int(np.ceil(shm_size / 4096) * 4096)

    args.shm_info = shm_info
    shm = SharedMemory(create=True, name=shm_info['name'], size=shm_size)
    data_array = np.ndarray(shape=shm_info['shape'], dtype=shm_info['data_type'], buffer=shm.buf)

    dir_path = f'{package_root}/data/evaluate_pid/gym_single'
    file_path = f'{dir_path}/0000.csv'
    preset_name = _get_preset_name(file_path)
    args.file_path = file_path
    args.preset_name = preset_name

    worker = Process(target=Analyzer, args=(args,))
    worker.start()
    worker.join()
    print(data_array)


if __name__ == '__main__':
    main()
