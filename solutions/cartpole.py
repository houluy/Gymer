import gym
import numpy as np
import tensorflow as tf
from itertools import product
from collections.abc import Sequence, MutableSequence, Iterable
from random import shuffle
tf.reset_default_graph()


class DQN:
    def __init__(self):
        self.ipt = tf.placeholder(name='input', shape=(None, 4), dtype=tf.float32)
        self.reward = tf.placeholder(name='reward', shape=(None, 1), dtype=tf.float32)
        self.action = tf.placeholder(name='action', shape=(None, 2), dtype=tf.float32)
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
        self.pool = [Experience(*tuple(0 for __ in range(self.attr_num))) for _ in range(self.size)]
        self.actions = []
        self.rewards = []
        self.observations = []
        self.n = []
        self.tg = [self.observations, self.actions, self.rewards, self.n]

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index):
        if isinstance(index, slice):
            self._shuffle()
            self._dist()
            return np.array(self.observations), np.array(self.actions), np.array(self.rewards), np.array(self.n)
        v = tuple(self.pool[index])
        return np.array(v)

    def __setitem__(self, index, value):
        self.pool[index] = Experience(*tuple(value))

    def __delitem__(self, index):
        del self.pool[index]

    def insert(self, index, value):
        e = Experience(*value)
        self.pool.insert(index, e)

    def _shuffle(self):
        shuffle(self.pool)

    def _dist(self):
        for p in self:
            for tg, item in zip(self.tg, p):
                tg.append(item)


class CartPoleQ:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.sess = tf.Session()
        self.actions = list(range(self.env.action_space.n))
        self.states = list(range(self.env.observation_space.shape[0]))
        self.Q = {
            k: np.random.random(1) for k in product(self.actions, self.states)
        }
        self.episodes = 10000
        self.update_paces = 20
        self.experience_size = 128
        self.epsilon = 0.5
        self.pool = ExperiencePool(self.experience_size)
        self.nets = DQN()

    def train(self):
        observation = self.env.reset()
        for episode in range(self.episodes):
            self.env.render()
            action = self.epsilon_greedy(observation)
            state = observation.copy()
            observation, reward, done, info = self.env.step(action)
            e = Experience(state, action, reward, observation)
            if done:
                observation = self.env.reset()
            else:
                e.reward += self.sess.run(self.nets.networks['target'], feed_dict={self.nets.ipt: state})
            self.pool.append((state, action, reward, observation))
            if episode % self.experience_size == 0:
                batch = self.pool[:]
                _, loss = self.sess.run([self.nets.train, self.nets.loss],
                                        feed_dict={
                                            self.nets.ipt: batch.observations,
                                            self.nets.action: batch.actions,
                                            self.nets.reward: batch.rewards
                                        })
        self.env.close()
        self.sess.close()

    def epsilon_greedy(self, observation):
        rand = np.random.random(1)
        if rand > self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.sess.run(self.nets.networks['target'], feed_dict={self.nets.ipt: observation}).argmax()


if __name__ == '__main__':
    c = CartPoleQ()
    print(c.Q)
    c.train()
