# -*- coding: utf-8 -*-
"""
Created on 2022/5/17 10:13

@author: qk
"""
import os
import numpy as np

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class PID(object):
    def __init__(self, kp, ki, kd, init_last_error, interval, limit=0.6):
        """
        Initialization.
        :param kp: Proportional Parameter
        :param ki: integral Parameter
        :param kd: Derivative Parameter
        """
        self.kp, self.ki, self.kd = kp, ki, kd  # PID参数
        self.last_error = init_last_error
        self.interval = interval

        self.i_error = 0.
        self.limit = limit

        self.reset()

    def reset(self):
        """
        Clear all parameters.
        """
        self.i_error = 0.

    def step(self, error):
        """
        State Update.
        :param error:
        """
        i_error = self.i_error + error
        d_error = (error - self.last_error) / self.interval
        output = self.kp * error + self.ki * i_error + self.kd * d_error
        output = np.clip(output, -self.limit, self.limit)
        self.update(error, i_error)
        # print(f'error: {error} i_error: {i_error} d_error: {d_error}')
        return output

    def update(self, error, i_error):
        self.last_error = error
        self.i_error = i_error


def main():
    pass


if __name__ == '__main__':
    main()
