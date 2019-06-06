import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

# from solutions.algorithms.ddpg import DeepDeterministicPolicyGradient


class Cartpole:
    def __init__(self):
        self.name = 'CartPole-v0'
        self.env = gym.make(self.name)
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.n

    def random_policy(self, state):
        return self.env.action_space.sample()

    def run(
        self,
        policy,
        episodes=100,
        info=False,
    ):
        for i in range(episodes):
            state = self.env.reset()
            self.env.render()
            done = False
            total_reward = 0
            while not done:
                action = self.action_wrapper(policy(state))
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                if info:
                    print(f'Info: state: {state}\n next_state: {next_state}\n action: {action}\n done: {done}\n')
                self.env.render()
            policy.stats(reward=total_reward)
        policy.show_reward()
        self.env.close()

    def __getattr__(self, key):
        return getattr(self.env, key)

    @staticmethod
    def action_wrapper(action):
        return 1 if action > 0.5 else 0

# class DQN:
#     def __init__(self):
#         self.ipt = tf.placeholder(name='input', shape=(None, 4), dtype=tf.float32)
#         self.reward = tf.placeholder(name='reward', shape=(None,), dtype=tf.float32)
#         self.action = tf.placeholder(name='action', shape=(None, 2), dtype=tf.int32)
#         self.gamma = 0.9
#         networks = ["target", "estimation"]
#         self.networks = {}
#         for n in networks:
#             weights1 = tf.get_variable(
#                 name=(n + 'weight'), shape=(4, 128), dtype=tf.float32,
#                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
#             )
#             weights2 = tf.get_variable(
#                 name=(n + 'weight2'), shape=(128, 2), dtype=tf.float32,
#                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)
#             )
#             bias = tf.get_variable(
#                 name=(n + 'bias'), shape=(128,), dtype=tf.float32,
#                 initializer=tf.constant_initializer(0)
#             )
#             es = tf.nn.relu(tf.matmul(self.ipt, weights1) + bias)
#             es = tf.nn.softmax(tf.matmul(es, weights2))
#             self.networks[n] = es
#
#         self.loss = tf.losses.mean_squared_error(
#             self.reward,
#             tf.gather_nd(self.networks["estimation"], self.action)
#         )
#         self.alpha = 0.3
#         self.train = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
#
#     @staticmethod
#     def copy_model(sess):
#         m1 = [t for t in tf.trainable_variables() if t.name.startswith('estimation')]
#         m1 = sorted(m1, key=lambda v: v.name)
#         m2 = [t for t in tf.trainable_variables() if t.name.startswith('target')]
#         m2 = sorted(m2, key=lambda v: v.name)
#
#         ops = []
#         for t1, t2 in zip(m1, m2):
#             ops.append(t2.assign(t1))
#         sess.run(ops)


# class CartPoleQ:
#     def __init__(self):
#         self.env = gym.make('CartPole-v0')
#         self.sess = tf.Session()
#         self.actions = list(range(self.env.action_space.n))
#         self.states = list(range(self.env.observation_space.shape[0]))
#         self.episodes = 10000
#         self.update_episodes = 20
#         self.save_episodes = 50
#         self.replay_episode = 128
#         self.experience_size = 256
#         self.pool = deque(maxlen=self.experience_size)
#         self.global_step = tf.Variable(0, trainable=False)
#         self.epsilon_decay = 0.9
#         self.epsilon_base = 0.9
#         self.epsilon_span = 100
#         self._epsilon = tf.train.exponential_decay(
#             self.epsilon_base,
#             self.global_step,
#             self.epsilon_span,
#             self.epsilon_decay,
#             staircase=True
#         )
#         self.gamma = 0.9
#
#     @property
#     def epsilon(self):
#         return self.sess.run(self._epsilon)
#
#     def train(self, show=False, load=False):
#         if load:
#             try:
#                 self.load()
#             except Exception as e:
#                 print(e)
#         observation = self.env.reset()
#         episodes = []
#         losses = []
#         persistence = 0
#         if show:
#             plt.figure('Loss')
#             plt.ion()
#         for episode in range(1, self.episodes + 1):
#             persistence += 1
#             self.env.render()
#             action = self.epsilon_greedy(observation)
#             state = observation.copy()
#             observation, reward, done, info = self.env.step(action)
#             if reward > 0:
#                 reward = 0.1
#             else:
#                 reward = -1.0
#             e = Experience(state, action, reward)
#             self.sess.run(tf.assign(self.global_step, episode))
#             if done:
#                 observation = self.env.reset()
#             else:
#                 target = self.sess.run(
#                     self.nets.networks['target'],
#                     feed_dict={self.nets.ipt: observation.reshape((1, 4))}
#                 )
#                 e.reward = e.reward + self.gamma * target.max()
#             self.pool.append(e)
#             count = episode % self.replay_episode
#             if count == 0:
#                 state_batch, action_batch, reward_batch = self.minibatch()
#                 _, loss = self.sess.run([self.nets.train, self.nets.loss],
#                                         feed_dict={
#                                             self.nets.ipt: state_batch,
#                                             self.nets.action: action_batch,
#                                             self.nets.reward: reward_batch
#                                         })
#                 episodes.append(episode)
#                 losses.append(loss)
#                 if show:
#                     self.show_loss(episodes, losses)
#             if episode % self.update_episodes:
#                 self.nets.copy_model(self.sess)
#             if episode % self.save_episodes:
#                 self.save()
#         if show:
#             plt.ioff()
#             plt.show()
#         self.env.close()
#         self.sess.close()
#
#     def test(self):
#         pass
#
#     def show_loss(self, episodes, losses):
#         plt.cla()
#         plt.title('Interactive loss and total reward over episodes')
#         plt.xlabel('Episodes')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.plot(episodes, losses, 'm+-', label='Instant loss')
#         # plt.plot(episodes, rewardarr, 'co-', label='Moving Average loss')
#         plt.legend()
#         plt.pause(0.01)
#
#     def minibatch(self):
#         state_batch = [0 for _ in range(self.experience_size)]
#         action_batch = [0 for _ in range(self.experience_size)]
#         reward_batch = [0 for _ in range(self.experience_size)]
#         for ind in range(self.experience_size):
#             e = random.choice(self.pool)
#             ie = iter(e)
#             state_batch[ind] = next(ie)
#             action = next(ie)
#             action_batch[ind] = [0, action]
#             reward_batch[ind] = next(ie)
#         return np.array(state_batch), np.array(action_batch), np.array(reward_batch)
#
#     def epsilon_greedy(self, observation):
#         rand = np.random.random(1)
#         if rand > self.epsilon:
#             return self.env.action_space.sample()
#         else:
#             return self.sess.run(self.nets.networks['target'],
#                                  feed_dict={self.nets.ipt: observation.reshape((1, 4))}).argmax()
#
#     def greedy_policy(self, observation):
#         return self.sess.run(self.nets.networks['target'],
#                              feed_dict={self.nets.ipt: observation.reshape((1, 4))}).argmax()
#

#
#     def save(self):
#         self.saver.save(self.sess, self.savefile)
#
#     def load(self):
#         self.saver.restore(self.sess, self.savefile)


if __name__ == '__main__':
    c = Cartpole()
    c.run()
