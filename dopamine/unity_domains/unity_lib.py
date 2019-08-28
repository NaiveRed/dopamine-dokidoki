# coding=utf-8
"""Unity-specific utilities including some network architectures.

* rainbow_network: Rainbow network without dueling and noisy net (same as atari_lib)
* rainbow_network_doki: Rainbow network wtih dueling and noisy net. (not complete)
* rainbow_mlp_doki: MLP version for vector observations

Modifying from atari_lib.py.
"""
import gin

import numpy as np
import tensorflow as tf

from dopamine.unity_domains.unity_wrappers import wrap_unity_env
from dopamine.unity_domains.noisy_net import noisy_linear_layer


@gin.configurable
def create_unity_environment(game_path=None):
    """Wrap the unity environment
  """
    assert game_path is not None and game_path[-4:] == ".exe", "Game path error."
    env = wrap_unity_env(game_path, port=9527, use_visual=False)
    return env


def nature_dqn_network(num_actions, network_type, state):
    """The convolutional network used to compute the agent's Q-values.

    Args:
        num_actions: int, number of actions.
        network_type: namedtuple, collection of expected values to return.
        state: `tf.Tensor`, contains the agent's current state.

    Returns:
        net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = tf.contrib.slim.conv2d(net, 32, [8, 8], stride=4)
    net = tf.contrib.slim.conv2d(net, 64, [4, 4], stride=2)
    net = tf.contrib.slim.conv2d(net, 64, [3, 3], stride=1)
    net = tf.contrib.slim.flatten(net)
    net = tf.contrib.slim.fully_connected(net, 512)
    q_values = tf.contrib.slim.fully_connected(net, num_actions, activation_fn=None)
    return network_type(q_values)


def rainbow_network(num_actions, num_atoms, support, network_type, state):
    """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
    weights_initializer = tf.contrib.slim.variance_scaling_initializer(factor=1.0 / np.sqrt(3.0),
                                                                       mode='FAN_IN',
                                                                       uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = tf.contrib.slim.conv2d(net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    net = tf.contrib.slim.conv2d(net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    net = tf.contrib.slim.conv2d(net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    net = tf.contrib.slim.flatten(net)
    net = tf.contrib.slim.fully_connected(net, 512, weights_initializer=weights_initializer)
    net = tf.contrib.slim.fully_connected(net,
                                          num_actions * num_atoms,
                                          activation_fn=None,
                                          weights_initializer=weights_initializer)

    logits = tf.reshape(net, [-1, num_actions, num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(support * probabilities, axis=2)
    return network_type(q_values, logits, probabilities)


@gin.configurable
def rainbow_doki_mlp(num_actions,
                     num_atoms,
                     support,
                     network_type,
                     state,
                     is_training_ph,
                     num_layer=2,
                     dueling=True,
                     noisy=True):
    """The convolutional network used to compute agent's Q-value distributions.

    Args:
        num_actions: int, number of actions.
        num_atoms: int, the number of buckets of the value function distribution.
        support: tf.linspace, the support of the Q-value distribution.
        network_type: namedtuple, collection of expected values to return.
        state: `tf.Tensor`, contains the agent's current state.
        is_training_ph: `tf.placeholder` with tf.bool type, indicate the training flag for noisy net.
        num_layer: int, number of hidden layers
        dueling: bool, enable dueling network.
        noisy: bool, enable noisy network.

    Returns:
        net: _network_type object containing the tensors output by the network.
    """

    net = tf.cast(state, tf.float32)
    net = tf.contrib.slim.flatten(net)
    for _ in range(num_layer - 1):
        net = tf.contrib.slim.fully_connected(net, 512)

    if dueling:

        # Value
        if noisy:
            value = noisy_linear_layer('noisy_val_1', net, [512, 512], is_training_ph)
            value = noisy_linear_layer('noisy_val_2', value, [512, num_atoms], is_training_ph)
        else:
            value = tf.contrib.slim.fully_connected(net, 512)
            value = tf.contrib.slim.fully_connected(value, num_atoms, activation_fn=None)

        # Advantage
        if noisy:
            adv = noisy_linear_layer('noisy_adv_1', net, [512, 512], is_training_ph)
            adv = noisy_linear_layer('noisy_adv_2', adv, [512, num_actions * num_atoms],
                                     is_training_ph)
        else:
            adv = tf.contrib.slim.fully_connected(net, 512)
            adv = tf.contrib.slim.fully_connected(adv, num_actions * num_atoms, activation_fn=None)

        value = tf.reshape(value, [-1, 1, num_atoms])
        adv = tf.reshape(adv, [-1, num_actions, num_atoms])
        logits = value + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
        probabilities = tf.nn.softmax(logits, axis=2)

    else:

        if noisy:
            net = noisy_linear_layer(net, [512, 512])
            net = noisy_linear_layer(net, [512, num_actions * num_atoms])
        else:
            net = tf.contrib.slim.fully_connected(net, 512)
            net = tf.contrib.slim.fully_connected(net, num_actions * num_atoms, activation_fn=None)

        logits = tf.reshape(net, [-1, num_actions, num_atoms])
        probabilities = tf.contrib.layers.softmax(logits)

    q_values = tf.reduce_sum(support * probabilities, axis=2)
    return network_type(q_values, logits, probabilities)
