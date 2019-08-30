# -*- coding:utf-8 -*-
# __author__ = 'shishuai.yan'

from __future__ import print_function, with_statement, absolute_import, division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.model import Model
import configs.global_variable as gl

class HCNet(Model):
    def __prelu(self, inputs):
        alpha = tf.get_variable('alpha', shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
        inputs = tf.nn.relu(inputs) - alpha * (tf.nn.relu(-inputs))
        return inputs

    def predict(self, inputs, is_training=True, name='HCNet'):
        # inputs = tf.transpose(inputs, (0, 2, 1, 3), name='HCNet/transpose')  # mtcnn处理图片格式为：[N, W, H, C]
        with tf.variable_scope(name, values=[inputs]):
            with slim.arg_scope([slim.conv2d], stride=1, padding='VALID',
                                activation_fn=self.__prelu,
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                # weights_regularizer=slim.l1_regularizer(0.0005)):
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME'):
                    # 48 * 48 * 3 -> 46 * 46 * 32       70 -> 68
                    inputs = slim.conv2d(inputs, 32, 3, scope='Conv2d_1_3x3')
                    # 46 * 46 * 32 -> 23 * 23 * 32      68 -> 34
                    inputs = slim.max_pool2d(inputs, 3, scope='MaxPool_1_3x3')
                    # 23 * 23 * 32 -> 21 * 21 * 64      34 -> 32
                    inputs = slim.conv2d(inputs, 64, 3, scope='Conv2d_2_3x3')
                    # 21 * 21 * 64 -> 10 * 10 * 64      32 -> 16
                    inputs = slim.max_pool2d(inputs, 3, scope='MaxPool_2_3x3', padding='VALID')
                    # 10 * 10 * 64 -> 8 * 8 * 64        16 -> 14
                    inputs = slim.conv2d(inputs, 64, 3, scope='Conv2d_3_3x3')
                    # 8 * 8 * 64 -> 4 * 4 * 64          14 -> 7
                    inputs = slim.max_pool2d(inputs, 2, scope='MaxPool_3_2x2')
                    # 4 * 4 * 64 -> 2 * 2 * 128         7 -> 5
                    inputs = slim.conv2d(inputs, 128, 3, scope='Conv2d_4_3x3')
                    # 3 * 3 * 128 -> 1 * 1 * 128        5 -> 1
                    inputs = slim.max_pool2d(inputs, 5, scope='MaxPool_4_2x2')
                    print(inputs)
                    # 1 * 1 * 128 -> 128
                    inputs = slim.flatten(inputs)
                    print(inputs)

                    classes_2 = slim.fully_connected(inputs, 2, scope='FC_5_classes_2', activation_fn=tf.nn.softmax)
                    classes_2 = tf.identity(classes_2, name=gl.output_tensor_name)
        return [classes_2]

    # def predict(self, inputs, is_training=True, name='HCNet'):
    #     # inputs = tf.transpose(inputs, (0, 2, 1, 3), name='HCNet/transpose')  # mtcnn处理图片格式为：[N, W, H, C]
    #     with tf.variable_scope(name, values=[inputs]):
    #         with slim.arg_scope([slim.conv2d], stride=1, padding='VALID',       # 指定covn默认padding为VALID
    #                             activation_fn=self.__prelu,
    #                             weights_initializer=slim.xavier_initializer(),
    #                             biases_initializer=tf.zeros_initializer(),
    #                             # weights_regularizer=slim.l1_regularizer(0.0005)):
    #                             weights_regularizer=slim.l2_regularizer(0.0005)):
    #             with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME'):
    #                 # 48 * 48 * 3 -> 46 * 46 * 32       60 -> 58
    #                 inputs = slim.conv2d(inputs, 32, 3, scope='Conv2d_1_3x3')   # 指定max_pool默认padding为SAME
    #                 # print(inputs)
    #                 # 46 * 46 * 32 -> 23 * 23 * 32      58 -> 29
    #                 inputs = slim.max_pool2d(inputs, 3, scope='MaxPool_1_3x3')
    #                 # print(inputs)
    #                 # 23 * 23 * 32 -> 21 * 21 * 64      29 -> 27
    #                 inputs = slim.conv2d(inputs, 64, 3, scope='Conv2d_2_3x3')
    #                 # print(inputs)
    #                 # 21 * 21 * 64 -> 10 * 10 * 64      27 -> 13
    #                 inputs = slim.max_pool2d(inputs, 3, scope='MaxPool_2_3x3')
    #                 # print(inputs)
    #                 # 10 * 10 * 64 -> 8 * 8 * 64        13 -> 11
    #                 inputs = slim.conv2d(inputs, 64, 3, scope='Conv2d_3_3x3')
    #                 # print(inputs)
    #                 # 8 * 8 * 64 -> 4 * 4 * 64          11 -> 5
    #                 inputs = slim.max_pool2d(inputs, 2, scope='MaxPool_3_2x2')
    #                 # print(inputs)
    #                 # 4 * 4 * 64 -> 2 * 2 * 128         5 -> 3
    #                 inputs = slim.conv2d(inputs, 128, 3, scope='Conv2d_4_3x3')
    #                 # print(inputs)
    #                 # 3 * 3 * 128 -> 1 * 1 * 128        5 -> 1
    #                 inputs = slim.max_pool2d(inputs, 4, scope='MaxPool_4_2x2', padding='VALID')
    #                 # print(inputs)
    #                 # 1 * 1 * 128 -> 128
    #                 inputs = slim.flatten(inputs)
    #
    #                 classes_2 = slim.fully_connected(inputs, 2, scope='FC_5_classes_2', activation_fn=tf.nn.softmax)
    #                 classes_2 = tf.identity(classes_2, name=gl.output_tensor_name)
    #     return [classes_2]
