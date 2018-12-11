import gym
import numpy as np
import tensorflow as tf
from collections.abc import Sequence, MutableSequence, Iterable
from random import shuffle
import matplotlib.pyplot as plt
tf.reset_default_graph()


class DQN:
    def __init__(self):
        self.ipt = tf.placeholder(name='input', shape=(None, 4), dtype=tf.float32)
        self.reward = tf.placeholder(name='reward', shape=(None,), dtype=tf.float32)
        self.action = tf.placeholder(name='action', shape=(None, 2), dtype=tf.int32)
        self.gamma = 0.9
        networks = ["target", "estimation"]
        self.networks = {}
        for n in networks:
            weights1 = tf.get_variable(
                name=(n + 'weight'), shape=(4, 128), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
            )
            weights2 = tf.get_variable(
                name=(n + 'weight2'), shape=(128, 2), dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
            )
            bias = tf.get_variable(
                name=(n + 'bias'), shape=(128,), dtype=tf.float32,
                initializer=tf.constant_initializer(0)
            )
            es = tf.nn.relu(tf.matmul(self.ipt, weights1) + bias)
            es = tf.nn.softmax(tf.matmul(es, weights2))
            self.networks[n] = es

        self.loss = tf.losses.mean_squared_error(
            self.reward,
            tf.gather_nd(self.networks["estimation"], self.action)
        )
        self.alpha = 0.3
        self.train = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)


class Experience(Iterable):
    __slots__ = ('observation', 'action', 'reward', 'n')

    def __init__(self, *args):
        self.observation, self.action, self.reward, self.n = tuple(args)

    def __iter__(self):
        yield self.observation
        yield self.action
        yield self.reward
        yield self.n


class ExperiencePool(MutableSequence):
    def __init__(self, size):
        self.size = size
        self.attr_num = 4
        self.pool = [Experience(*tuple(0 for _ in range(self.attr_num))) for x in range(size)]
        self.actions = [0 for _ in range(size)]
        self.rewards = [0 for _ in range(size)]
        self.observations = [0 for _ in range(size)]
        self.n = [0 for _ in range(size)]
        self.tg = [self.observations, self.actions, self.rewards, self.n]

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index):
        if isinstance(index, slice):
            self._shuffle()
            self._dist()
            return np.array(self.observations).reshape((self.size, 4)),\
                   np.array(self.actions).reshape((self.size, 2)),\
                   np.array(self.rewards).reshape((self.size,)),\
                   np.array(self.n).reshape((self.size, 4))
        v = tuple(self.pool[index])
        return np.array(v)

    def __setitem__(self, index, value):
        value[1] = [0, value[1]]
        self.pool[index] = Experience(*tuple(value))

    def __delitem__(self, index):
        del self.pool[index]

    def insert(self, index, value):
        value[1] = [0, value[1]]
        e = Experience(*value)
        self.pool.insert(index, e)

    def _shuffle(self):
        shuffle(self.pool)

    def _dist(self):
        for ind, p in enumerate(self):
            for tg, item in zip(self.tg, p):
                tg[ind] = [item]


class CartPoleQ:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.sess = tf.Session()
        self.actions = list(range(self.env.action_space.n))
        self.states = list(range(self.env.observation_space.shape[0]))
        self.episodes = 10000
        self.update_paces = 20
        self.experience_size = 128
        self.epsilon = 0.5
        self.pool = ExperiencePool(self.experience_size)
        self.nets = DQN()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        observation = self.env.reset()
        episodes = []
        endepisodes = []
        losses = []
        rewardarr = []
        total_reward = 0
        persistence = 0
        plt.figure('Loss')
        plt.ion()
        for episode in range(1, self.episodes + 1):
            persistence += 1
            self.env.render()
            action = self.epsilon_greedy(observation)
            state = observation.copy()
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            e = Experience(state, action, reward, observation)
            if done:
                observation = self.env.reset()
                rewardarr.append(total_reward / persistence)
                persistence = 0
                endepisodes.append(episode)
            else:
                e.reward += self.sess.run(self.nets.networks['target'], feed_dict={self.nets.ipt: state.reshape((1, 4))})
            count = episode % self.experience_size
            self.pool[count] = [state, action, reward, observation]
            if count == 0:
                batch = self.pool[:]
                _, loss = self.sess.run([self.nets.train, self.nets.loss],
                                        feed_dict={
                                            self.nets.ipt: batch[0],
                                            self.nets.action: batch[1],
                                            self.nets.reward: batch[2]
                                        })
                episodes.append(episode)
                #rewardarr.append(total_reward)
                losses.append(loss)
                plt.cla()
                plt.title('Interactive loss and total reward over episodes')
                plt.xlabel('Episodes')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.plot(endepisodes, rewardarr, 'm+-', label='Instant loss')
                #plt.plot(episodes, rewardarr, 'co-', label='Moving Average loss')
                plt.legend()
                plt.pause(0.01)
            #if episode %
        plt.ioff()
        plt.show()
        self.env.close()
        self.sess.close()

    def epsilon_greedy(self, observation):
        rand = np.random.random(1)
        if rand > self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.sess.run(self.nets.networks['target'],
                                 feed_dict={self.nets.ipt: observation.reshape((1, 4))}).argmax()


if __name__ == '__main__':
    c = CartPoleQ()
    c.train()
