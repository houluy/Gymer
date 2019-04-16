import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class Algo:
    def __init__(
        self,
        pool_size,
        batch_size,
        info_moment,
        save_round,
        train_round,
    ):
        self.lossarr = np.array([])
        self.rewardarr = np.array([])
        self.pool_size = pool_size
        self.experience_pool = deque(maxlen=pool_size)
        self.train_round = train_round
        self.batch_size = batch_size
        self.info_moment = info_moment
        self.save_round = save_round

    def __str__(self):
        return self.__class__.__name__

    def show(self):
        plt.figure(1)
        plt.plot(self.lossarr, label='algo:{}'.format(str(self)))
        plt.legend()
        plt.title('Loss of NN')
        plt.figure(2)
        plt.plot(self.rewardarr, label='algo:{}'.format(str(self)))
        plt.legend()
        plt.title('Average reward of RL solution')
        plt.show()
