# -*- coding:utf-8 -*-
# __author__ = 'shishuai.yan'

from __future__ import print_function, with_statement, absolute_import, division
import tensorflow as tf
import configs.global_variable as gl

class Model(object):
    def predict(self, inputs, is_training=True, default_name='HCNet'):
        pass
        # with tf.variable_scope(None, default_name, [inputs]):
        # return head_sex, head_age, head_hat, head_glass

    def loss(self, prediction, groundtruth):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction[0], labels=groundtruth))  # np.log(np.exp(a)/(np.exp(a)+np.exp(b)))
        tf.summary.scalar('loss', loss)

        return loss

    def acc(self, prediction, groundtruth):
        acc = tf.reduce_mean(tf.cast(tf.equal(self.postprocess(prediction[0]), groundtruth), 'float'))
        tf.summary.scalar('acc', acc)

        acc_list = [acc]
        return acc_list

    def postprocess(self, prediction):
        # prediction = tf.nn.softmax(prediction, name='prediction')# 将hcff网络返回的值转为百分比，[N, ]
        classes = tf.cast(tf.argmax(prediction, axis=1), dtype=tf.int32)# hcff网络返回的是[N, 2]的数据，分别代表
        return classes

