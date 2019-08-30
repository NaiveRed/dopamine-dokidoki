# coding=utf-8
"""
Noisy Networks for Exploration.
https://arxiv.org/abs/1706.10295
"""
import numpy as np
import tensorflow as tf


def mu_variable(name, shape):
    """\mu is sampled from independent uniform distribution [-1/sqrt(p), 1/sqrt(p)]
    (exclude 1/sqrt(p) in this implementation). p is the number of inputs.

    Args:
        name: Variable name, can not be same as existing variables.
        shape: `list` or array-type, variable shape e.g. [input_unit, output_unit].
    """
    return tf.compat.v1.get_variable(name=name,
                                     initializer=tf.random.uniform(shape,
                                                                   minval=-1.0 / np.sqrt(shape[0]),
                                                                   maxval=1.0 / np.sqrt(shape[0]),
                                                                   dtype=tf.float32),
                                     trainable=True)


def sigma_variable(name, shape, sigma_0=0.5):
    """
    Args:
        name: Variable name, can not be same as existing variables.
        shape: `list` or array-type, variable shape e.g. [input_unit, output_unit].
        sigma_0: The hyperparameter for initialization, default is 0.5 in paper.
    """
    return tf.compat.v1.get_variable(name=name,
                                     initializer=tf.constant(sigma_0 / np.sqrt(shape[0]),
                                                             shape=shape,
                                                             dtype=tf.float32),
                                     trainable=True)


def noisy_linear_layer(name, input_, shape, is_training_ph):
    """
    Args:
        name: Variable name, can not be same as existing variables.
        input_: Input tensor.
        shape: `list` or array-type, variable shape e.g. [input_unit, output_unit].
        is_training_ph: `tf.placeholder` with tf.bool type, indicate the training flag for noisy net.
    """
    def factorised_fn(x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def factorised_noise():
        """
        Returns:
            (weight noise, bias noise): Factorised Gaussian noise for weights and bias.
        """
        eps_i = tf.random.normal([shape[0]])  # p unit for noise of the inputs
        eps_j = tf.random.normal([shape[1]])  # q unit for noise of the output
        f_i = factorised_fn(eps_i)
        f_j = factorised_fn(eps_j)
        eps_w = tf.tensordot(f_i, f_j, axes=0)  # Outer product, [shape[0], shape[1]]
        eps_b = f_j
        return eps_w, eps_b

    with tf.compat.v1.variable_scope(name):
        # Weight variables
        mu_weight = mu_variable('mu_weight', shape)
        sigma_weight = sigma_variable('sigma_weight', shape)

        # Bias variables
        mu_bias = mu_variable('mu_bias', [shape[1]])
        sigma_bias = sigma_variable('sigma_bias', [shape[1]])

    eps_w, eps_b = tf.cond(is_training_ph, factorised_noise,
                           (lambda: (tf.zeros(shape), tf.zeros([shape[1]]))))

    linear_weight = mu_weight + sigma_weight * eps_w  # Element-wised product
    linear_bias = mu_bias + sigma_bias * eps_b  # Element-wised product

    return tf.matmul(input_, linear_weight) + linear_bias
