# -*- coding:utf-8 -*-
# __author__ = 'shishuai.yan'


# Note: tensorboard --logdir=xxx --host=127.0.0.1

import os, platform

image_size = 64
batch_size = 128
epoch_num = 2000
learning_rate = 0.0001
input_name = 'HCNet/inputs_clear'
label_name = 'HCNet/labels_clear'
optimizer = 'ADAM'  # ADADELTA, ADAM, SGD
train_valid_dict = {'train': {0: r'D:\Desktop\shishuai.yan\Desktop\git_code\tfClassifier\image_classification\images\0',
                              1: r'D:\Desktop\shishuai.yan\Desktop\git_code\tfClassifier\image_classification\images\1'},
                        'valid': {0: r'D:\Desktop\shishuai.yan\Desktop\temp\0', 1: r'D:\Desktop\shishuai.yan\Desktop\temp\1'}}
ckpt_save_dir = r'D:\Desktop\shishuai.yan\Desktop\temp\out'




# if platform.system() == 'Windows':
#     class_size = 32
# else:
#     class_size = 256
# epoch_num = 2000
#
# if platform.system() == 'Windows':
#     # base_dir = '//192.168.16.42/share/haihong.qin/darknet/trainval/structure_sub_imgs/all'
#     base_dir = '//192.168.16.42/share/haihong.qin/darknet/trainval/structure_sub_imgs/all/clear_side_img'
# else:
#     # base_dir = '/home/hostfs/haihong.qin/darknet/trainval/structure_sub_imgs/all'
#     base_dir = '/home/hostfs/haihong.qin/darknet/trainval/structure_sub_imgs/all/clear_side_img'
# train_csv_path = os.path.join(base_dir, 'clear_80.csv') # structure_csv_95.csv
# valid_csv_path = os.path.join(base_dir, 'clear_20.csv')  # structure_csv_5.csv
# if platform.system() == 'Windows':
#     valid_size = 10
# else:
#     valid_size = 500    # 实际: *18
# # ckpt_save_dir = os.path.join(base_dir, 'hcff_output', 'ckpt_multi_temp')     # ------------------
# ckpt_save_dir = os.path.join(base_dir, 'hcff_output', 'ckpt_clear_mobilenet')     # ------------------
# if not os.path.isdir(ckpt_save_dir):
#     os.mkdir(ckpt_save_dir)
# input_name = 'HCNet/inputs_clear'     # ------------------
# label_name = 'HCNet/labels_clear'    # ------------------
# output_tensor_name = 'head_classes_2'   # 实际Tensor名：HCNet/head_classes_2:0
#
# # --------------------------------- Evaluate --------------------------------------
# # 加__为内部变量，外部无法应用
# __divided_save_dir = os.path.join(ckpt_save_dir, 'divided_img')
# __folder_names = [['clear', 'blurry']]
#
# tensor_name_dict = {'input': 'HCNet/inputs_clear:0', 'head_clear': 'HCNet/head_classes_2:0'}
# # tensor_name_dict = {'input': 'HCNet/inputs_multi:0', 'head_sex': 'None/head_sex:0', 'head_age': 'None/head_age:0', 'head_hat': 'None/head_hat:0', 'head_glass': 'None/head_glass:0'}
# ckpt_path = os.path.join(ckpt_save_dir, 'model_1999_350-701649')
# divided_save_dir_list = [[os.path.join(__divided_save_dir, folder) for folder in folder_name] for folder_name in __folder_names]
# for dir_list in divided_save_dir_list:
#     for dir in dir_list:
#         if not os.path.isdir(dir):
#             os.makedirs(dir)