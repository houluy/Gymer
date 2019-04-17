import matplotlib.pyplot as plt
# import numpy as np
from collections import deque


class Algo:
    def __init__(
        self,
        env,
        pool_size,
        batch_size,
        info_moment,
        save_round,
        train_round,
        render=True,
    ):
        self.lossarr = []
        self.rewardarr = []
        self.pool_size = pool_size
        self.experience_pool = deque(maxlen=pool_size)
        self.train_round = train_round
        self.batch_size = batch_size
        self.info_moment = info_moment
        self.save_round = save_round
        self.env = env
        self.state_shape = self.env.state_shape
        self.action_shape = self.env.action_shape
        self._env_info()
        if render:
            self.render = self.env.render
        else:
            self.render = lambda: None

    def __str__(self):
        return self.__class__.__name__

    def _env_info(self):
        print(
            'Environment info:\nName: {}\nState shape: {}\n Action shape: {}'.format(
            self.env.name,
            self.state_shape,
            self.action_shape,
        ))

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


class Experience:
    __slots__ = ('state', 'action', 'reward', 'next_state', 'done')

    def __init__(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

