# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 19:15
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : eval_on_dataset.py
# @Software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from model.mini_vgg_test import mini_VGG
from keras.preprocessing.image import ImageDataGenerator

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    image_data = ImageDataGenerator(rotation_range=0.2,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale=1.0 / 255)

    # 构造验证集数据生成器
    #dataset_dir = os.path.join(cfg.dataset_dir, "train")
    dataset_dir = os.path.abspath("./data/cifar10/val")
    image_data = ImageDataGenerator(rescale=1.0 / 255)
    datagen = image_data.flow_from_directory(dataset_dir,
                                             class_mode='categorical',
                                             batch_size = 1,
                                             target_size=(32,32),
                                             shuffle=False)

    # 初始化相关参数
    iter_num = datagen.samples       # 训练集1个epoch的迭代次数
    label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # 初始化VGG16，并进行训练
    model_path = os.path.abspath("./checkpoints/20200530195246/stage2-trained-model.h5")
    mini_vgg = mini_VGG(model_path=model_path)
    print(mini_vgg.eval_generator(datagen,iter_num,label_names))

if __name__ == '__main__':
    run_main()