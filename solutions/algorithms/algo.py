import matplotlib.pyplot as plt
# import numpy as np
from collections import deque, namedtuple
from collections.abc import MutableSequence
import tensorflow as tf


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
        self.regularizer_weight = 0.03
        self.dropout_rate = 0.5
        if render:
            self.render = self.env.render
        else:
            self.render = lambda: None
        tf.reset_default_graph()
        self.sess = tf.Session()

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
                clayer = tf.nn.dropout(clayer, rate=self.dropout_rate)
        return clayer

    @staticmethod
    def gen_weights(
        model_name,
        scope_name,
        shape,
        bias_shape,
        stddev=.1,
        bias=.1,
        regularizer=None,
        wl=None
    ):
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

    def __str__(self):
        return self.__class__.__name__

    def _env_info(self):
        print(
            'Environment info:\nName: {}\nState shape: {}\n Action shape: {}'.format(
                self.env.name,
                self.state_shape,
                self.action_shape,
            )
        )

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

    def __del__(self):
        try:
            self.sess.close()
        except AttributeError:
            print('Something wrong before the session was created')


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

FCLayer = namedtuple('FCLayer', (
'name', 'layer', 'shape', 'regularizer', 'activation'
))

__all__ = ['FCLayer', 'Experience', 'Algo']
