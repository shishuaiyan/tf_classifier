import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class GenerateData:
    def __init__(self, train_valid_dict, image_size=64, batch_size=128, epoch=2, valid_batch_size=2000):
        self.train_paths, self.train_labels = self.__get_path_label(train_valid_dict['train'])
        self.valid_paths, self.valid_labels = self.__get_path_label(train_valid_dict['valid'])
        self.image_size = image_size
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.epoch = epoch

    def __get_path_label(self, img_dir_dict):
        '''
        :param img_dir_dict: {'img_dir':label}
        :return:
        '''
        labels = []
        img_paths = []
        for label in img_dir_dict:
            label = int(label)
            img_path_list = os.listdir(img_dir_dict[label])
            img_path_list = [os.path.join(img_dir_dict[label], img_name) for img_name in img_path_list]
            img_paths += img_path_list
            labels += [label] * len(img_path_list)
        return img_paths, labels

    def __preprocess(self, img_path, training):
        image_raw = tf.io.read_file(img_path)
        image_raw = tf.image.decode_png(image_raw, channels=3)
        image = tf.cast(image_raw, tf.float32)
        image = image / 255
        if training:
            image = tf.image.resize(image, (round(self.image_size*1.1), round(self.image_size*1.1)))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_crop(image, (self.image_size, self.image_size, 3))
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_hue(image, max_delta=0.1)
        else:
            image = tf.image.resize(image, (self.image_size, self.image_size))
        return image

    def __preprocess_train(self, img_path, label):
        image = self.__preprocess(img_path, training=True)
        return image, label

    def __preprocess_valid(self, img_path, label):
        image = self.__preprocess(img_path, training=False)
        return image, label

    def get_data(self, training):
        labels = self.train_labels if training else self.valid_labels
        paths = self.train_paths if training else self.valid_paths
        preprocess_fun = self.__preprocess_train if training else self.__preprocess_valid
        batch_size = self.batch_size if training else self.valid_batch_size

        assert (len(labels) == len(paths))
        train_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        train_dataset = train_dataset.cache()   # 据说能加速
        train_dataset = train_dataset.shuffle(len(labels) + 100)
        if training:
            train_dataset = train_dataset.repeat(self.epoch)  # 若不填参数，无限循环
        # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.map(preprocess_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)

        return train_dataset


def main():
    tf.enable_eager_execution() # 开启动态图

    train_valid_dict = {'train': {0: r'D:\Desktop\shishuai.yan\Desktop\temp\0', 1: r'D:\Desktop\shishuai.yan\Desktop\temp\1'},
                        'valid': {0: r'D:\Desktop\shishuai.yan\Desktop\temp\0', 1: r'D:\Desktop\shishuai.yan\Desktop\temp\1'}}
    gd = GenerateData(train_valid_dict, image_size=64, batch_size=2)
    dataset = gd.get_data(training=True)

    print(gd.train_paths)
    print(gd.train_labels)
    # print(train_dataset.take(1))
    for j, (batch_img, batch_label) in enumerate(dataset):
        print('batch_num: ', j)
        for i in range(batch_img.shape[0]):
            # print(batch_img[i])
            print(batch_img[i].shape, batch_label[i])
            plt.imshow(batch_img[i])
            plt.show()

if __name__ == '__main__':
    main()