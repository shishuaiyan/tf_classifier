# -*- coding:utf-8 -*-
# __author__ = 'shishuai.yan'

import tensorflow as tf
from models.model import Model

def create_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)

# def batchnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
#     return tf.layers.batch_normalization(inputs, epsilon=epsilon, momentum=momentum, training=is_training)

def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter2d = create_variable('filter', shape=[filter_size, filter_size, in_channels, channel_multiplier], initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.depthwise_conv2d(inputs, filter2d, strides=[1, strides, strides, 1], padding='SAME', rate=[1,1])

def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter2d = create_variable('filter', shape=[filter_size, filter_size, in_channels, num_filters], initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.conv2d(inputs, filter2d, strides=[1, strides, strides, 1], padding="SAME")

def avg_pool(inputs, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding="VALID")

def fc(inputs, n_out, scope, use_bias=True):
    inputs_shape = inputs.get_shape().as_list()
    n_in = inputs_shape[-1]
    with tf.variable_scope(scope):
        weight = create_variable('weight', shape=[n_in, n_out], initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:
            bias = create_variable('bias', shape=[n_out], initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)
        return tf.matmul(inputs, weight)


class MobileNet(Model):

    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier, scope, strides, is_training):
        num_filters = round(num_filters*width_multiplier)
        with tf.variable_scope(scope):
            dw_conv = depthwise_conv2d(inputs, 'depthwise_conv', strides=strides)
            dw_conv = tf.layers.batch_normalization(dw_conv, training=is_training)
            dw_conv = tf.nn.relu(dw_conv)
            pw_conv = conv2d(dw_conv, 'pointwise_conv', num_filters)
            pw_conv = tf.layers.batch_normalization(pw_conv, training=is_training)
            return tf.nn.relu(pw_conv)

    def predict(self, inputs, is_training=False, scope='mobileNet', width_multiplier=1):
        with tf.variable_scope(scope):
            net = conv2d(inputs, 'conv_1', round(32*width_multiplier), filter_size=3, strides=2)
            net = tf.nn.relu(tf.layers.batch_normalization(net, training=is_training))
            net = self._depthwise_separable_conv2d(net, 64, width_multiplier, 'ds_conv_2', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 128, width_multiplier, 'ds_conv_3', strides=2, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 128, width_multiplier, 'ds_conv_4', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 256, width_multiplier, 'ds_conv_5', strides=2, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 256, width_multiplier, 'ds_conv_6', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 512, width_multiplier, 'ds_conv_7', strides=2, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 512, width_multiplier, 'ds_conv_8', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 512, width_multiplier, 'ds_conv_9', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 512, width_multiplier, 'ds_conv_10', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 512, width_multiplier, 'ds_conv_11', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 512, width_multiplier, 'ds_conv_12', strides=1, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 1024, width_multiplier, 'ds_conv_13', strides=2, is_training=is_training)
            net = self._depthwise_separable_conv2d(net, 1024, width_multiplier, 'ds_conv_14', strides=1, is_training=is_training)
            net = avg_pool(net, 2, 'avg_pool_15')
            net = tf.squeeze(net, [1,2], name='SpatialSqueeze')
            logits = fc(net, 2, 'fc_16')
            predictions = tf.nn.softmax(logits, name='predictions')
            return [predictions]
