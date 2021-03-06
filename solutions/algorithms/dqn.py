import tensorflow as tf
from collections import namedtuple, deque
import numpy as np
import random
import matplotlib.pyplot as plt
from solutions.algorithms.algo import *


class DQN(Algo):
    name2layer = {
        'FC': FCLayer,
        'Con': ConvLayer,
    }

    def __init__(
        self,
        env,
        typ='FC',
        layers=[4, 8, 4],
        activations=(tf.nn.relu, tf.nn.relu, tf.nn.relu, None),
        stddev=5e-2,
        biases=0.1,
    ):
        super().__init__(
            env=env,
        )
        self.ipt = tf.placeholder(tf.float32, shape=(None, self.state_shape))
        self.action = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.float32, [None])
        self.reg_lambda = 0.03
        self.alpha = 0.3
        self.gamma = 0.99
        self.reward = 0
        self.total_q = 0
        self.total_loss = 0
        self.train_step = 0  # The step of training process
        self.episode = 0  # The round of episodic game
        self.epoch = 0  # The number of epochs in one episode (one game)
        self.epochs = []  # Epoch list
        self.run_step = 0  # The total step from the very beginning
        # run_step = sum(epochs) = train_step + batch_size

        self.stddev = stddev
        self.biases = biases
        layers.append(self.action_shape)
        layer_params = {
            'typ': typ,
            'layers': layers,
            'stddev': self.stddev,
            'bias': self.biases,
            'regularizer': True,
            'regularizer_weight': self.reg_lambda,
            'activation': activations,
        }
        self.Q_model = self.build_model(self.ipt, self._define_layers('Q', layer_params.copy()))
        self.target = self.build_model(self.ipt, self._define_layers('target', layer_params.copy()))
        self.model = lambda state: self.sess.run(self.Q_model, feed_dict={self.ipt: state})  # This is used for consistency in Algo __call__
        self._define_loss()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self._copy_weights('Q', 'target')
        # self.merged = tf.summary.merge_all()
        self.train_writter = tf.summary.FileWriter(r"C:\Users\houlu\PycharmProjects\GYMER\board_logs\train", self.sess.graph)

    def _define_layers(self, name, layer_params):
        layers = layer_params.pop('layers')
        typ = layer_params.pop('typ')
        activation = layer_params.pop('activation')
        return [
            self.name2layer[typ](
                name=name,
                layer=ind,
                shape=shape,
                activation=activation[ind],
                **layer_params
            ) for ind, shape in enumerate(layers)
        ]

    def _define_loss(self):
        # Convert action to one hot vector
        a_one_hot = tf.one_hot(self.action, self.action_shape, 1.0, 0.0)
        self.q_value = tf.reduce_sum(
            tf.multiply(
                self.Q_model,
                a_one_hot
            ),
            reduction_indices=1
        )

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(self.y - self.q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Q_value", self.q_value)

    def one_train(self):
        train_start = False
        action = self(self.state, train=True)
        observation, reward, done, info = self.env.step(action)
        self.reward += reward
        done = 1 if done else 0
        self.experience_pool.append(
            Experience(
                self.state,
                action,
                reward,
                observation,
                done,
            )
        )
        self.run_step += 1
        self.epoch += 1
        if self.experience_size < self.pool_size:  # Max number of experiences
            self.experience_size += 1
        if self.experience_size >= self.batch_size:  # Ready to train the networks
            train_start = True
            batch = random.sample(self.experience_pool, self.batch_size)
            batch = self._convert(batch)
            loss, q, _ = self.sess.run(
                [self.loss, self.q_value, self.optimizer],
                feed_dict={
                    self.ipt: batch['state'],
                    self.action: batch['action'],
                    self.y: batch['reward']
                }
            )
            self.total_q += q.mean()
            self.total_loss += loss
            # Update the target network
            if not (self.train_round % self.update_round):
                self._update_target()
            # Update exploration
            if not (self.train_round % self.epsilon_round):
                self._update_epsilon()
            if not (self.train_round % self.info_round):
                self._show_info()
                self._tensorboard(batch, self.train_round // self.info_round)
            self.train_round += 1
        if done:
            if train_start:
                self.average_q.append(self.total_q / self.epoch)
                self.total_q = 0
                self.average_loss.append(self.total_loss / self.epoch)
                self.total_loss = 0
            self.episode += 1
            self.epochs.append(self.epoch)
            self.state = self.env.reset()
            self.rewards.append(self.reward / self.epoch)
            self.epoch = 0
            self.reward = 0
            if self.debug:
                assert self.run_step == sum(self.epochs)
        else:
            self.state = observation
        if self.debug:
            if self.train_round > 0:
                assert self.run_step == self.train_round + self.batch_size - 1
        # if not (self.train_round % self.save_round):
        #     self.saver.save(self.sess, str(self.save_file))
        # if not (self.train_round % self.stats_round):
        #     self._stats()

    def _update_target(self):
        self._copy_weights("Q", "target")

    def train(self, show=False):
        # try:
        #     self.saver.restore(self.sess, self.save_file)
        # except ValueError:
        #     print('First-time train')
        # except tf.errors.InvalidArgumentError:
        #     print('New game')
        # except tf.errors.DataLossError:
        #     print('FATAL ERROR, start new game')
        # except tf.errors.NotFoundError:
        #     print('New game')
        self.state = self.env.reset()
        for episode in range(self.total_round):
            if show:
                self.env.render()
            self.one_train()
        self._show_info()
        self.show_q()
        self.show_loss()
        self.show_rewards()
        # self.saver.save(self.sess, str(self.save_file))
        #     epoch = 0
        #     ep_reward = 0
        #     total_loss = 0
        #     total_q = 0
        #
        #         ep_reward += reward
        #         self.experience_pool.append(Experience(
        #             state,
        #             action,
        #             reward,
        #             next_state,
        #             done,
        #         ))  # Gaining experience pool
        #         self.experience_size += 1
        #         if show:
        #             self.env.render()
        #         loss = None
        #         if self.experience_size >= self.batch_size:  # Until it satisfy minibatch size
        #             minibatch = random.sample(self.experience_pool, self.batch_size)
        #             #self.sess.run(tf.assign(self.global_step, episode))
        #             batch = self._convert(minibatch)
        #             _, loss, q = self.sess.run(
        #                 [self.optimizer, self.loss, self.q_value],
        #                 feed_dict={
        #                     self.ipt: batch['state'],
        #                     self.action: batch['action'],
        #                     self.y: batch['reward']
        #                 }
        #             )
        #             total_loss += loss
        #             update += 1
        #             if update > self.update_round:
        #                 update = 0
        #                 self._copy_weights("Q", "target")
        #             q = q.max()
        #             total_q += q
        #         if not (epoch % 5):
        #             print(f'Current epoch {epoch} in episode {episode}, current loss: {loss}')
        #         epoch += 1
        #     else:
        #         self.lossarr.append(total_loss/epoch)
        #         self.ep_rewards.append(ep_reward)
        #         self.q_v.append(total_q)
        #     if not (episode % self.info_moment):
        #         print(f'Current training episode: {episode}')
        #     if not (episode % self.save_round):
        #         self.saver.save(self.sess, str(self.save_file))
        # self.show_loss()
        # self.show_q()
        # self.show_reward()
        # self.env.close()

    def _convert(self, minibatch):
        'Convert minibatch from namedtuple to multi-dimensional matrix'
        batch = {
            'state': [],
            'reward': [],
            'action': [],
        }
        for block in minibatch:
            batch['state'].append(block.state)
            batch['action'].append(block.action)
            if block.done:
                reward = block.reward
            else:
                target = self.sess.run(self.target, feed_dict={self.ipt: np.array([block.next_state])})
                reward = block.reward + self.gamma * target.max()
            batch['reward'].append(reward)
        batch['action'] = np.array(batch['action']).T
        batch['reward'] = np.array(batch['reward']).T
        return batch

    def _tensorboard(self, minibatch, i):
        loss = self.sess.run([self.loss], feed_dict={
                    self.ipt: minibatch['state'],
                    self.action: minibatch['action'],
                    self.y: minibatch['reward']
                })
        self.train_writter.add_summary(loss, i)

    # @staticmethod
    # def gen_weights(scope_name, shape, bias_shape, stddev=.1, bias=.1, regularizer=None, wl=None):
    #     weight_init = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
    #     bias_init = tf.constant_initializer(bias)
    #     weights = tf.get_variable('{}-weights'.format(scope_name), shape=shape, initializer=weight_init)
    #     biases = tf.get_variable('{}-biases'.format(scope_name), shape=bias_shape, initializer=bias_init)
    #     if regularizer is not None:
    #         weights_loss = tf.multiply(tf.nn.l2_loss(weights), wl, name='weights-loss')
    #         tf.add_to_collection('losses', weights_loss)
    #     return weights, biases

    # def _build_layer(self, ipt_layer, opt_layer):
    #     clayer = ipt_layer
    #     with tf.variable_scope(opt_layer.name, reuse=tf.AUTO_REUSE):
    #         if isinstance(opt_layer, ConvLayer):
    #             weight_shape = [*opt_layer.kernel, opt_layer.channels, opt_layer.number]
    #             weights, biases = self.gen_weights(
    #                 opt_layer.name + str(opt_layer.layer),
    #                 weight_shape,
    #                 bias_shape=[opt_layer.number],
    #                 stddev=opt_layer.stddev,
    #                 bias=opt_layer.bias,
    #             )
    #             clayer = tf.nn.conv2d(clayer, weights, strides=opt_layer.strides, padding='SAME')
    #             clayer = tf.nn.relu(tf.nn.bias_add(clayer, biases))
    #         elif isinstance(opt_layer, PoolLayer):
    #             clayer = tf.nn.max_pool(clayer, ksize=opt_layer.ksize, strides=opt_layer.strides, padding='SAME')
    #         elif isinstance(opt_layer, FCLayer):
    #             clayer = tf.layers.Flatten()(clayer)
    #             ipt_size = clayer.get_shape()[-1]
    #             weight_shape = [ipt_size, opt_layer.shape]
    #             weights, biases = self.gen_weights(
    #                 opt_layer.name + str(opt_layer.layer),
    #                 weight_shape,
    #                 bias_shape=[opt_layer.shape],
    #                 regularizer=opt_layer.regularizer,
    #                 wl=opt_layer.regularizer_weight,
    #             )
    #             clayer = tf.add(tf.matmul(clayer, weights), biases)
    #             if opt_layer.activation is not None:
    #                 clayer = opt_layer.activation(clayer)
    #     return clayer
    #
    # def build_all(self, structure):
    #     current = self.ipt
    #     layer = None
    #     for layer in structure:
    #         current = self._build_layer(current, layer)
    #     if layer.name == 'Q':
    #         current = tf.layers.Flatten()(current)
    #         return tf.matmul(current, self.mask)
    #     else:
    #         return current


# class ConvolutionNetwork(QApproximation):
#     def __init__(
#         self,
#
#     ):
#         self.ipt_shape = ipt_shape
#         self.ipt_channel = ipt_channel
#         self.batch_size = batch_size
#         self.opt_size = out_size
#         self.ipt = tf.placeholder(tf.float32, shape=(None, *self.ipt_shape, self.ipt_channel))
#         self.reward = tf.placeholder(tf.float32, shape=(None, 1))
#         self.mask = tf.placeholder(tf.float32, shape=(self.opt_size, None))
#
#         # Below is the output mask that only the chosen neural (action) will output Q.
#         # self.opt_mask = tf.placeholder(tf.float32, shape=(self.opt_size, None))
#         self.reg_lambda = 0.03
#         self.alpha = 0.3
#
#         c_strides = (1, 2, 2, 1)
#         p_strides = (1, 2, 2, 1)
#         k_size = (1, 3, 3, 1)
#         stddev = 5e-2
#         biases = 0.1
#         model_name = ['Q', 'target']
#         self.networks = {}
#         for name in model_name:
#             self.networks[name] = self.build_all([
#                 ConvLayer(
#                     name=name,
#                     layer=1,
#                     kernel=(3, 3),
#                     strides=c_strides,
#                     number=16,
#                     channels=self.ipt_channel,
#                     stddev=stddev,
#                     bias=biases
#                 ),
#                 # self.PoolLayers(
#                 #     name=name,
#                 #     layer=2,
#                 #     ksize=k_size,
#                 #     strides=p_strides,
#                 # ),
#                 ConvLayer(
#                     name=name,
#                     layer=2,
#                     kernel=(3, 3),
#                     strides=c_strides,
#                     number=32,
#                     channels=16,
#                     stddev=stddev,
#                     bias=biases
#                 ),
#                 # self.PoolLayers(
#                 #     name=name,
#                 #     layer=4,
#                 #     ksize=k_size,
#                 #     strides=p_strides,
#                 # ),
#                 # self.ConvLayers(
#                 #     name=name,
#                 #     layer=5,
#                 #     kernel=(1, 1),
#                 #     strides=c_strides,
#                 #     number=128,
#                 #     channels=32,
#                 #     stddev=stddev,
#                 #     bias=biases,
#                 # ),
#                 # self.ConvLayers(
#                 #     name=name,
#                 #     layer=6,
#                 #     kernel=(1, 1),
#                 #     strides=c_strides,
#                 #     number=self.opt_size,
#                 #     channels=128,
#                 #     stddev=stddev,
#                 #     bias=biases,
#                 # )
#                 # self.FCLayers(
#                 #     name=name,
#                 #     layer=5,
#                 #     shape=128,
#                 #     stddev=stddev,
#                 #     bias=biases,
#                 #     regularizer=True,
#                 #     regularizer_weight=self.reg_lambda,
#                 #     activation=tf.nn.relu,
#                 # ),
#                 FCLayer(
#                     name=name,
#                     layer=3,
#                     shape=self.opt_size,
#                     stddev=stddev,
#                     bias=biases,
#                     regularizer=None,
#                     regularizer_weight=self.reg_lambda,
#                     activation=None,
#                 )
#             ], name=name)
#         self._action = 0
#         self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
#         self.saver = tf.train.Saver()


class DQN2:

    exp = namedtuple('exp', ('state', 'action', 'instant', 'next_state', 'terminal'))

    def __init__(self, game):
        self.game = game
        self.ipt_size, self.opt_size = self.game.size
        self.q_network = DQN(self.ipt_size, self.opt_size, batch_size=128)
        self.global_step = tf.Variable(0, trainable=False)
        self.epsilon_decay = 0.9
        self.epsilon_base = 0.9
        self.epsilon_span = 200
        self._epsilon = tf.train.exponential_decay(
            self.epsilon_base,
            self.global_step,
            self.epsilon_span,
            self.epsilon_decay,
            staircase=True
        )
        self.actions = list(range(self.opt_size))
        self.exp_size = 1000
        self.experience_pool = deque(maxlen=self.exp_size)
        self.episodes = 200
        self.minibatch_size = 128
        self.target_update_episode = 10
        self.save_episode = 100
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.model_file = '../models/model.ckpt'
        self.hyper_params = {
            'gamma': 0.9,
        }

    # def gain_experiences(self, game):
    #     for _ in range(self.experience_size):
    #         state = self.observe(game)
    #         action_index = np.random.choice(self.actions)
    #         game.interact(action_index)
    #         reward = self.q.reward(game)
    #         next_state = self.observe(game)
    #         self.experience_pool.append(self.exp(
    #             state=state,
    #             action=action_index,
    #             instant=reward,
    #             next_state=next_state,
    #         ))
    #         if reward == -10:
    #             game.reset()

    @property
    def epsilon(self):
        return self.sess.run(self._epsilon)

    def train(self, window=None):
        try:
            self.q_network.saver.restore(self.sess, self.model_file)
        except ValueError:
            print('First-time train')
        except tf.errors.InvalidArgumentError:
            print('New game')
        except tf.errors.DataLossError:
            print('FATAL ERROR, start new game')
        except tf.errors.NotFoundError:
            print('New game')
        self.game.reset()
        # plt.figure('Loss')
        # plt.ion()
        self.lossarr = []
        lossave = []
        self.experience_size = 0
        episode = 0
        while episode < self.episodes:
            state = self.game.state
            while True:
                state = state.reshape((self.ipt_size, self.ipt_size, 1), order='F')
                epsilon = np.random.rand()
                action = self.epsilon_greedy(epsilon, state)
                next_state, reward, terminal, _ = self.game.step(action)
                next_state = next_state.reshape((self.ipt_size, self.ipt_size, 1), order='F')
                if reward > 0:
                    self.game.new_food()
                self.experience_pool.append(self.exp(
                    state=state,
                    action=action,
                    instant=reward,
                    next_state=next_state,
                    terminal=terminal,
                ))  # Gaining experience pool
                self.experience_size += 1
                self.game.render(window)
                if self.experience_size >= self.minibatch_size:  # Until it satisfy minibatch size
                    minibatch = random.sample(self.experience_pool, self.minibatch_size)
                    episode += 1
                    self.sess.run(tf.assign(self.global_step, episode))
                    batch = self._convert(minibatch)
                    _, loss = self.sess.run(
                        [self.optimizer, self.q_network.loss],
                        feed_dict={
                            self.ipt: batch['state'],
                            self.mask: batch['action'],
                            self.reward: batch['reward'],
                        }
                    )
                    self.lossarr.append(loss)
                    #lossave.append(sum(lossarr)/(episode + 1))
                    #self.show(episodes, lossarr, lossave)
                    if episode % self.target_update_episode == 0:
                        self.q_network._copy_model(self.sess)
                    if episode % self.save_episode == 0:
                        self.q_network.saver.save(self.sess, self.model_file)
                if terminal:
                    break
            self.game.reset()
        self.show_loss()
        self.game.close(window)

    def show(self, episodes, lossarr, lossave):
        plt.cla()
        plt.title('Interactive loss over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.plot(episodes, lossarr, 'm+-', label='Instant loss')
        plt.plot(episodes, lossave, 'co-', label='Moving Average loss')
        plt.legend()
        plt.pause(0.01)

    def show_loss(self):
        plt.figure()
        plt.plot(self.lossarr, 'm+-', label='Loss')
        plt.legend()
        plt.show()

    def epsilon_greedy(self, epsilon, state):
        if epsilon < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.greedy(state)

    def greedy(self, state):
        return self.sess.run(self.q, feed_dict={self.ipt: [state], self.mask: [[1.], [1.], [1.], [1.]]}).argmax()

    @property
    def target(self):
        return self.q_network['target']

    @property
    def q(self):
        return self.q_network['Q']

    @property
    def ipt(self):
        return self.q_network.ipt

    @property
    def reward(self):
        return self.q_network.reward

    @property
    def optimizer(self):
        return self.q_network.optimizer

    @property
    def mask(self):
        return self.q_network.mask

    def __getattr__(self, name):
        if name in self.hyper_params:
            return self.hyper_params[name]
        else:
            raise AttributeError()

    def __del__(self):
        self.sess.close()
