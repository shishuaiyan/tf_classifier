from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import configs.global_variable as gl
import cv2, os
from data.generate_data import GenerateData
import numpy as np

def ckpt2pb(ckpt_path, pb_save_dir, output_node_name_list):
    from tensorflow.python.framework import graph_util
    with tf.Session() as sess:
        # Load .ckpt file
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        # Save as .pb file
        graph_def = tf.get_default_graph().as_graph_def()
        # convert_variables_to_constants函数，会将计算图中的变量取值以常量的形式保存。
        # 在保存模型文件的时候，我们只是导出了GraphDef部分，GraphDef保存了从输入层到输出层的计算过程。
        # 在保存的时候，通过convert_variables_to_constants函数来指定保存的节点名称而不是张量的名称，“add:0”是张量的名称而"add"表示的是节点的名称。
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_node_name_list)
        if ckpt_path[-4:] == 'ckpt':
            pb_save_path = '{}.pb'.format(ckpt_path[:ckpt_path.rfind('.')])
        else:
            pb_save_path = '{}.pb'.format(ckpt_path)
        print('pb model save path:', pb_save_path)
        with tf.gfile.GFile(os.path.join(pb_save_dir, pb_save_path), 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)
def test_ckpt2pb():
    output_node_name_list =['mobileNet/predictions']
    ckpt_path = r'D:\Desktop\shishuai.yan\Desktop\temp\out\model-34'
    pb_save_dir = r'D:\Desktop\shishuai.yan\Desktop\temp'
    ckpt2pb(ckpt_path, pb_save_dir, output_node_name_list)
# test_ckpt2pb()


class evaluate_pb:
    def __init__(self, pb_path):
        with tf.gfile.FastGFile(pb_path, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(self.graph_def, name='')
        self.sess = tf.Session()
        self.predictions = self.sess.graph.get_tensor_by_name('mobileNet/predictions:0')

    def look_nodes(self):
        for i,n in enumerate(self.graph_def.node):
            print("Name of the node - %s" %(n.name))

    def eval_img(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255
        image = cv2.resize(image, (64, 64))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        print(image)
        predictions = self.sess.run(self.predictions, feed_dict={'HCNet/inputs_clear:0': image})
        return predictions[0][0]
def test_evaluate_pb():
    eval = evaluate_pb(r'D:\Desktop\shishuai.yan\Desktop\temp\model-34.pb')
    # eval.look_nodes()
    print(eval.eval_img(r'D:\Desktop\shishuai.yan\Desktop\working\1.jpg'))
test_evaluate_pb()

def evaluate_ckpt():
    gd = GenerateData(gl.train_valid_dict, gl.image_size, gl.batch_size, gl.epoch_num)
    valid_data = gd.get_data(training=False)

    ckpt = tf.train.Checkpoint()
    ckpt.restore(tf.train.latest_checkpoint())

