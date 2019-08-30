from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import platform, os

import configs.global_variable as gl
from data.generate_data import GenerateData
from models.mobilenetv1 import MobileNet

# tf.data的使用: https://blog.csdn.net/lyb3b3b/article/details/82910863
#               https://blog.csdn.net/z2539329562/article/details/80753355

class Trainer:

    def __init__(self):
        self.model = MobileNet()
        self.gd = GenerateData(gl.train_valid_dict, gl.image_size, gl.batch_size, gl.epoch_num)
        train_dataset = self.gd.get_data(training=True)
        valid_dataset = self.gd.get_data(training=False)
        self.train_data_iterator = train_dataset.make_initializable_iterator()
        self.valid_data_iterator = valid_dataset.make_initializable_iterator()
        self.learning_rate = gl.learning_rate
        optimizer = gl.optimizer
        if optimizer == 'ADADELTA':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif optimizer == 'ADAM':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        ckpt_dir = gl.ckpt_save_dir
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def train(self):
        inputs = tf.placeholder(tf.float32, shape=[None, gl.image_size, gl.image_size, 3], name=gl.input_name) # input normalized data -> tf.float32; unnormalized data -> tf.uint8
        labels = tf.placeholder(tf.int32, shape=[None], name=gl.label_name)
        predictions = self.model.predict(inputs, is_training=True)
        loss = self.model.loss(predictions, labels)
        acc_list = self.model.acc(predictions, labels)
        optimizer = tf.train.AdamOptimizer(gl.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = optimizer.minimize(loss, global_step=global_step)


        train_next_element = self.train_data_iterator.get_next()
        valid_next_element = self.valid_data_iterator.get_next()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(gl.ckpt_save_dir, sess.graph)
            merged = tf.summary.merge_all()     # 合并summary
            saver = tf.train.Saver(max_to_keep=5)

            sess.run(tf.global_variables_initializer())
            sess.run(self.valid_data_iterator.initializer)
            sess.run(self.train_data_iterator.initializer)

            valid_images, valid_labels = sess.run(valid_next_element)
            while 1:
                batch_images, batch_labels = sess.run(train_next_element)
                # print(batch_labels.dtype)
                train_dict = {inputs:batch_images, labels:batch_labels}
                sess.run(train_step, feed_dict=train_dict)
                loss_, acc_list_, result_, step = sess.run([loss, acc_list, merged, global_step], feed_dict=train_dict)
                writer.add_summary(result_, step)
                saver.save(sess, os.path.join(gl.ckpt_save_dir, 'model'), global_step=step)
                if sum(acc_list_)/len(acc_list_) > 0.75:   # 0.75
                    valid_dict = {inputs: valid_images, labels: valid_labels}
                    valid_predictions = sess.run(predictions, feed_dict=valid_dict)
                    # valid_acc_list_ = calValidAcc(valid_predictions, valid_batch_label)
                    print('task: {:<6s} step: {:<5d}  loss: {:<1.10f}   acc: {}   valid_acc: {}'.format('multi', step, loss_, acc_list_, '999'))
                else:
                    print('task: {:<6s} step: {:<5d}  loss: {:<1.10f}   acc: {}'.format('multi', step, loss_, acc_list_))
            writer.close()

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    main()