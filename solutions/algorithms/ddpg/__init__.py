import tensorflow as tf
from itertools import chain
from collections import namedtuple
from collections.abc import MutableSequence
import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib

from solutions.algorithms.ddpg.noise import OUNoise
from solutions.algorithms.algo import Algo, Experience

FCLayer = namedtuple('FCLayer', (
'name', 'layer', 'shape', 'regularizer', 'activation'
))


class DeepDeterministicPolicyGradient(Algo):
    def __init__(
        self,
        env,
        batch_size=128,
        pool_size=1000,
        info_moment=100,
        save_round=100,
        train_round=2000,
    ):
        super().__init__(
            env=env,
            pool_size=pool_size,
            batch_size=batch_size,
            train_round=train_round,
            info_moment=info_moment,
            save_round=save_round,
        )
        tf.reset_default_graph()
        self.reward_shape = 1
        self.critic_opt_shape = 1
        self.dropout = 0.5
        self.actor_learning_rate = 0.01
        self.critic_learning_rate = 0.01
        self.actor_decay = 0.01
        self.action_typ = 3
        self.critic_decay = 0.01
        self.critic_Q_input = tf.placeholder(tf.float32, shape=(None, 1))
        self.regularizer_weight = 0.03
        self.tau = 0.001
        self.gamma = 0.9
        self.max_value = 0

        self.actor_input = tf.placeholder(tf.float32, shape=(None, self.state_shape))
        actor_layers_dict = {}
        for name in ['actor', 'target_actor']:
            actor_layers_dict[name] = [
                FCLayer(name=name, layer=1, shape=16, regularizer=True, activation=tf.nn.relu),
                FCLayer(name=name, layer=2, shape=32, regularizer=True, activation=tf.nn.relu),
                FCLayer(name=name, layer=3, shape=16, regularizer=True, activation=tf.nn.relu),
                FCLayer(name=name, layer=4, shape=self.action_shape, regularizer=False, activation=None)
            ]
        self.actor_layers = actor_layers_dict['actor']
        self.actor_target_layers = actor_layers_dict['target_actor']
        del actor_layers_dict
        self.critic_state_input = tf.placeholder(
            tf.float32,
            shape=(
                None,
                self.state_shape
            )
        )
        self.critic_action_input = tf.placeholder(
            tf.float32,
            shape=(
                None,
                self.action_shape
            )
        )
        critic_layers_dict = {}
        for name in ['critic', 'target_critic']:
            critic_layers_dict[name] = [
                FCLayer(name=name, layer=1, shape=16, regularizer=True, activation=tf.nn.relu),
                FCLayer(name=name, layer=2, shape=32, regularizer=True, activation=tf.nn.relu),
                FCLayer(name=name, layer=3, shape=16, regularizer=True, activation=tf.nn.relu),
                FCLayer(name=name, layer=4, shape=self.critic_opt_shape, regularizer=False, activation=None)
            ]
        self.critic_layers = critic_layers_dict['critic']
        self.critic_target_layers = critic_layers_dict['target_critic']
        self.actor_model = self.build_model(
            self.actor_input,
            self.actor_layers,
        )
        self.critic_model = self.build_model(
            [self.critic_state_input, self.critic_action_input],
            self.critic_layers,
        )
        self.actor_target_model = self.build_model(
            self.actor_input,
            self.actor_target_layers,
        )
        self.critic_target_model = self.build_model(
            [self.critic_state_input, self.critic_action_input],
            self.critic_target_layers,
        )
        self.sess = tf.Session()
        self.critic_loss = tf.reduce_mean(tf.square(self.critic_Q_input - self.critic_model))
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.critic_loss + sum(tf.get_collection('critic-losses')))

        self.critic_gradient = tf.gradients(self.critic_model, self.critic_action_input)

        # self.episode_pool = []  # Save experiences of each episodes

        self.exploration_noise = OUNoise(self.action_shape)

        self.q_gradient_input = tf.placeholder(tf.float32, shape=[None, self.action_shape])
        self.actor_parameters = list(chain(*tf.get_collection('actor')))
        self.critic_parameters = list(chain(*tf.get_collection('critic')))
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        self.actor_target_update = self.ema.apply(self.actor_parameters)
        self.critic_target_update = self.ema.apply(self.critic_parameters)
        self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters, - self.q_gradient_input)
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(
            zip(self.parameters_gradients, self.actor_parameters))
        self.sess.run(tf.global_variables_initializer())
        self.critic_loss_record = []
        self.actor_j_record = []
        self._copy_weights('critic', 'target_critic')
        self._copy_weights('actor', 'target_actor')
        self.saver = tf.train.Saver()
        self.save_file = pathlib.Path(__file__).parent.parent.parent / pathlib.Path('models/cartpole/ddpg/model')
        try:
            self.saver.restore(self.sess, str(self.save_file))
        except ValueError:
            print('First-time train')
        except tf.errors.InvalidArgumentError:
            print('New game')
        except tf.errors.DataLossError:
            print('FATAL ERROR, start new game')
        except tf.errors.NotFoundError:
            print('New game')

    def _build_layer(self, ipt_layer, opt_layer):
        with tf.variable_scope(opt_layer.name, reuse=tf.AUTO_REUSE):
            ipt_layer = tf.layers.Flatten()(ipt_layer)
            ipt_size = ipt_layer.get_shape()[-1]
            weight_shape = [ipt_size, opt_layer.shape]
            weights, biases = self.gen_weights(
                opt_layer.name,
                opt_layer.name + str(opt_layer.layer),
                weight_shape,
                bias_shape=[opt_layer.shape],
                regularizer=opt_layer.regularizer,
                wl=self.regularizer_weight,
            )
            tf.add_to_collections(opt_layer.name, [weights, biases])
            clayer = tf.add(tf.matmul(ipt_layer, weights), biases)
            if opt_layer.activation is not None:
                clayer = opt_layer.activation(clayer)
                clayer = tf.nn.dropout(clayer, rate=self.dropout)
        return clayer

    @staticmethod
    def gen_weights(model_name, scope_name, shape, bias_shape, stddev=.1, bias=.1,
                    regularizer=None, wl=None):
        weight_init = tf.truncated_normal_initializer(dtype=tf.float32,
                                                      stddev=stddev)
        bias_init = tf.constant_initializer(bias)
        weights = tf.get_variable('{}-weights'.format(scope_name),
                                  shape=shape, initializer=weight_init)
        biases = tf.get_variable('{}-biases'.format(scope_name),
                                 shape=bias_shape, initializer=bias_init)
        if regularizer is not None:
            weights_loss = tf.multiply(tf.nn.l2_loss(weights), wl,
                                       name='weights-loss')
            tf.add_to_collection('{}-losses'.format(model_name), weights_loss)
        return weights, biases

    def build_model(self, ipt, layers):
        if isinstance(ipt, MutableSequence):
            current = tf.concat(ipt, axis=1)
        else:
            current = ipt
        for layer in layers:
            current = self._build_layer(current, layer)
        return current

    def _copy_weights(self, src_name, dest_name):
        m1 = [t for t in tf.trainable_variables() if t.name.startswith(src_name)]
        m1 = sorted(m1, key=lambda v: v.name)
        m2 = [t for t in tf.trainable_variables() if t.name.startswith(dest_name)]
        m2 = sorted(m2, key=lambda v: v.name)

        ops = []
        for t1, t2 in zip(m1, m2):
            ops.append(t2.assign(t1))
        self.sess.run(ops)

    def __del__(self):
        try:
            self.sess.close()
        except AttributeError:
            print('Something wrong before the session was created')

    def _batch_process(self, batch):
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_next_state = []
        for exp in batch:
            batch_state.append(exp.state)
            batch_next_state.append(exp.next_state)
            batch_action.append(exp.action)
            batch_reward.append(exp.reward)
            batch_done.append(exp.done)
        return np.array(batch_state),\
            np.array(batch_action),\
            np.array(batch_reward),\
            np.array(batch_next_state),\
            np.array(batch_done)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def train(self):
        size = 0
        for e in range(self.train_round):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_wrapper(self(state, train=True))
                next_state, reward, done, info = self.env.step(action)
                self.experience_pool.append(Experience(state, action, reward, next_state, done))
                size += 1
                if size > self.batch_size:
                    loss = self.train_with_batch(random.sample(self.experience_pool, self.batch_size))
                    self.lossarr.append(loss)
                state = next_state
                if not (e % self.info_moment):
                    print(f'Current training round: {e}')
                if not (e % self.save_round):
                    print(self.save_file)
                    self.saver.save(self.sess, str(self.save_file))

    def train_with_batch(self, batch):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self._batch_process(batch)
        batch_next_action = self.sess.run(self.actor_target_model, feed_dict={
            self.actor_input: batch_next_state,
        })

        batch_target_q = self.sess.run(self.critic_target_model, feed_dict={
            self.critic_state_input: batch_next_state,
            self.critic_action_input: batch_next_action,
        })

        y = []
        for ind in range(self.batch_size):
            if batch_done[ind]:
                y.append(batch_reward[ind])
            else:
                y.append(batch_reward[ind] + self.gamma * batch_target_q[ind])
        y = np.resize(y, [self.batch_size, 1])
        _, critic_loss = self.sess.run(
            [self.critic_optimizer, self.critic_loss],
            feed_dict={
                self.critic_state_input: batch_state,
                self.critic_action_input: batch_action,
                self.critic_Q_input: y
            }
        )

        # Update gradient
        batch_action = self.sess.run(self.actor_model, feed_dict={
            self.actor_input: batch_state,
        })

        q_gradients_batch = self.sess.run(self.critic_gradient, feed_dict={
            self.critic_state_input: batch_state,
            self.critic_action_input: batch_action,
        })[0]

        # Update actor model
        self.sess.run(self.actor_optimizer, feed_dict={
            self.q_gradient_input: q_gradients_batch,
            self.actor_input: batch_state
        })

        # Update target model
        self.sess.run([self.actor_target_update, self.critic_target_update])
        return critic_loss

    def __call__(self, state, train=False):
        action = self.sess.run(self.actor_model, feed_dict={
            self.actor_input: state.reshape((1, self.state_shape))
        })
        if train:
            return (action + self.exploration_noise.noise()).item()
        else:
            return action.item()

    def noise_action(self, state):
        return self(state) + self.exploration_noise.noise()


if __name__ == '__main__':
    ddpg = DeepDeterministicPolicyGradient(5, 5)

