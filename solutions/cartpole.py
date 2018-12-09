import gym
import numpy as np
import tensorflow as tf
from itertools import product
from collections.abc import MutableSequence
from random import shuffle
tf.reset_default_graph()


class DQN:
    def __init__(self):
        self.ipt = tf.placeholder(name='input', shape=(None, 4), dtype=tf.float32)
        self.reward = tf.placeholder(name='reward', shape=(None, 1), dtype=tf.float32)
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
            es = tf.nn.relu(tf.matmul(weights1, self.ipt) + bias)
            es = tf.nn.softmax(tf.matmul(weights2, es))
            self.networks[n] = es
        self.loss = tf.losses.mean_squared_error(
            self.reward + self.gamma * self.networks["target"].max(),
            self.networks["estimation"].max()
        )
        self.alpha = 0.3
        self.train = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)


class Experience:
    __slots__ = ('observation', 'reward', 'done', 'n')
    def __init__(self, *args):
        self.observation, self.reward, self.done, self.n = args


class ExperiencePool(MutableSequence):
    def __init__(self, size):
        self.size = size
        self.pool = [0 for _ in range(self.size)]

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index):
        if len(self) >= self.size:
            self._shuffle()
        v = self.pool[index]
        del self[index]
        return np.array(v)

    def __setitem__(self, index, value):
        self.pool[index] = Experience(value)

    def __delitem__(self, index):
        del self.pool[index]

    def insert(self, index, value):
        self.pool.insert(index, Experience(value))

    def _shuffle(self):
        shuffle(self.pool)

    def batch(self, target, sess):
        for e in self:
            if e


class CartPoleQ:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.sess = tf.Session()
        self.actions = list(range(self.env.action_space.n))
        self.states = list(range(self.env.observation_space.shape[0]))
        self.Q = {
            k:np.random.random(1) for k in product(self.actions, self.states)
        }
        self.episodes = 10000
        self.update_paces = 20
        self.experience_size = 128
        self.epsilon = 0.5
        self.pool = ExperiencePool(self.experience_size)

    def train(self):
        observation = self.env.reset()
        for episode in range(self.episodes):
            self.env.render()
            action = 0
            state = observation.copy()
            observation, reward, done, info = self.env.step(action)
            self.pool.append((state, reward, done, observation))
            if done:
                self.env.reset()
            if episode % self.experience_size == 0:
                batch = self.pool[:]


        self.env.close()
        self.sess.close()

    def epsilon_greedy(self):


if __name__ == '__main__':
    c = CartPoleQ()
    print(c.Q)
    c.train()
