# -*- coding: utf-8 -*-
# @Time    : 2020/5/24 22:53
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : train_mini_vgg.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from config.config import config
from model.mini_vgg_train import mini_VGG
from keras.preprocessing.image import ImageDataGenerator

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    batch_size = 128
    epoch = 50
    cfg = config(epoch = epoch,batch_size=batch_size)

    # 构造训练集和测试集数据生成器
    train_dataset_dir = os.path.abspath("./data/cifar10/train")
    val_dataset_dir = os.path.abspath("./data/cifar10/val")
    image_data =  ImageDataGenerator(rotation_range=0.2,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale= 1.0/255)
    train_datagen = image_data.flow_from_directory(train_dataset_dir,
                                                   class_mode='categorical',
                                                   batch_size = batch_size,
                                                   target_size=(32,32),
                                                   shuffle=True)
    val_datagen = image_data.flow_from_directory(val_dataset_dir,
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 target_size=(32,32),
                                                 shuffle=True)
    train_iter_num = train_datagen.samples // batch_size
    val_iter_num = val_datagen.samples // batch_size
    if train_datagen.samples % batch_size != 0:
        train_iter_num += 1
    if val_datagen.samples % batch_size != 0:
        val_iter_num += 1

    # 初始化VGG16，并进行测试批量图像
    mini_vgg = mini_VGG(cfg)
    mini_vgg.train(train_datagen=train_datagen,
                    val_datagen=val_datagen,
                    train_iter_num=train_iter_num,
                    val_iter_num=val_iter_num)

if __name__ == '__main__':
    run_main()