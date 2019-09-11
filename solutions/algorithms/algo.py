import matplotlib.pyplot as plt
import numpy as np
from collections import deque, namedtuple
from collections.abc import MutableSequence
import tensorflow as tf
import pathlib


ConvLayer = namedtuple('ConvLayer',
    ('name', 'layer', 'kernel', 'strides', 'number', 'channels', 'stddev', 'bias')
)
PoolLayer = namedtuple('PoolLayer', ('name', 'layer', 'ksize', 'strides'))
# Local Response Normalizations
# LRNLayers = namedtuple('LRNLayer', ('layer', 'type', 'radius', 'bias', 'alpha', 'beta'))
FCLayer = namedtuple('FCLayer',
    ('name', 'layer', 'shape', 'stddev', 'bias', 'regularizer', 'regularizer_weight', 'activation')
)


class Algo:
    def __init__(
        self,
        env,
        info_round=500,
        save_round=2000,
        update_round=4,
        epsilon_round=1000,
        total_round=3000,
        batch_size=1024,
        pool_size=4096,
        render=True,
        debug=True,
    ):
        self.pool_size = pool_size
        self.experience_pool = deque(maxlen=pool_size)
        self.total_round = total_round
        self.batch_size = batch_size
        self.info_round = info_round
        self.save_round = save_round
        self.stats_round = 5
        self.update_round = update_round  # Number of rounds to update parameters of target networks
        self.epsilon = 0.5
        self.epsilon_step = 0.02
        self.epsilon_round = epsilon_round  # Number of rounds to update epsilon
        self.env = env
        self.state = self.env.reset()
        self.state_shape = self.env.state_shape
        self.action_shape = self.env.action_shape
        self.model = lambda x: x  # This must be customized in subclass
        self._env_info()
        self.average_q = []
        self.average_loss = []
        self.average_reward = []
        self.rewards = []
        self.episode = 0
        self.epoch = 0
        self.run_step = 0
        self.train_round = 0
        self.experience_size = 0
        self.regularizer_weight = 0.03
        self.dropout_rate = 0.9
        self.debug = debug
        #self.global_step = tf.Variable(0, trainable=False)
        if render:
            self.render = self.env.render
        else:
            self.render = lambda: None
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.save_file = f'solutions/models/{self.env.name}/{str(self)}/model.ckpt'

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
        weight_name = f'{scope_name}-weights'
        bias_name = f'{scope_name}-biases'
        weights = tf.get_variable(weight_name,
                                  shape=shape, initializer=weight_init)
        biases = tf.get_variable(bias_name,
                                 shape=bias_shape, initializer=bias_init)
        tf.summary.histogram(weight_name, weights)
        tf.summary.histogram(bias_name, biases)
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

    def stats(
        self,
        **kwargs,
    ):
        for key, val in kwargs.items():
            t = getattr(self, key + 'arr', [])
            t.append(val)

    def _update_epsilon(self):
        epsilon = self.epsilon - self.epsilon_step
        if epsilon > 0:
            self.epsilon = epsilon

    @property
    def rp(self):
        return np.random.rand(self.action_shape)

    def __str__(self):
        return self.__class__.__name__

    # def __call__(self, state, train=False):
    #     raw_action = self.model(state.reshape((1, self.state_shape)))
    #     ret = raw_action if not train else raw_action + self.rp
    #     return ret.argmax()

    def __call__(self, state, train=False):
        randnum = np.random.rand()
        if not train or randnum > self.epsilon:
            action = self.model(state.reshape(1, self.state_shape)).argmax()
        else:
            action = self.env.random_policy()
        return action

    def _env_info(self):
        print(
            'Environment info:\nName: {}\nState shape: {}\n Action shape: {}'.format(
                self.env.name,
                self.state_shape,
                self.action_shape,
            )
        )

    def _show_info(self):
        print(
            f'Training info:\n',
            f'Run step: {self.run_step}\n'
            f'Episode: {self.episode},\n'
            f'Epoch: {self.epoch}\n'
            f'Train step: {self.train_round}\n'
            f'Epsilon: {self.epsilon}\n'
        )

    def show_loss(self):
        plt.figure(1)
        plt.plot(self.average_loss, label='algo:{}'.format(str(self)))
        plt.legend()
        plt.title('Loss of NN')
        plt.show()

    def show_q(self):
        plt.figure()
        plt.plot(self.average_q, label='algo: {}'.format(str(self)))
        plt.legend()
        plt.title('Q value')
        plt.show()

    def show_rewards(self):
        plt.figure(2)
        plt.plot(self.rewards, label='algo:{}'.format(str(self)))
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.legend()
        plt.title('Reward of each episode')
        plt.show()

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
        try:
            self.env.close()
        except AttributeError:
            print('ERROR')


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


__all__ = ['ConvLayer', 'PoolLayer', 'FCLayer', 'Experience', 'Algo']
