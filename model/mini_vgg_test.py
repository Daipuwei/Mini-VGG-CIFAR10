# -*- coding: utf-8 -*-
# @Time    : 2020/5/24 17:11
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : mini_vgg_test.py
# @Software: PyCharm

import os
import numpy as np
from sklearn.metrics import classification_report

from keras import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization

class mini_VGG(object):

    _default_dict_ = {
        "input_image_shape": (32,32,3),
        "model_path": os.path.abspath("./data/mini_vgg.h5")
    }

    def __init__(self,**kwargs):
        """
        这是VGG16的初始化函数
        :param kwargs: 参数字典
        """
        # 初始化相关参数
        self.__dict__.update(self._default_dict_)
        self.__dict__.update(kwargs)

        # 加载模型
        try:
            self.model = load_model(self.model_path)
        except:
            self.build_model()      # 搭建MiNi-VGG
            self.model.load_weights(self.model_path,by_name=True,skip_mismatch=True)
        print("loads model from: ",self.model_path)

    def build_model(self):
        """
        这是Mini-VGG网络的搭建函数
        :return:
        """
        # 初始化网络输入
        self.image_input = Input(shape=self.input_image_shape,name="image_input")

        y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(self.image_input)
        y = BatchNormalization()(y)
        y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
        y = Dropout(0.25)(y)

        y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

        y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
        y = Dropout(0.25)(y)

        y = Flatten()(y)
        y = Dense(512, activation='relu', kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        y = Dense(10, activation='softmax', kernel_initializer='he_normal')(y)

        self.model = Model(self.image_input,y,name="Mini-VGG")
        self.model.summary()

    def eval_generator(self,datagen,iter_num,label_names):
        """
        这是利用数据集生成器对模型进行评估的函数
        :param datagen: 数据集生成器
        :param iter_num: 数据集生成器迭代次数
        :param label_names: 标签名称
        :return:
        """
        y_real = []
        y_pred = []
        for i in np.arange(iter_num):
            batch_images,batch_real_labels = datagen.__next__()
            y_real.append(np.argmax(batch_real_labels,axis=-1))
            batch_pred_labels = self.model.predict_on_batch(batch_images)
            y_pred.append(np.argmax(batch_pred_labels,axis=-1))
        y_real = np.concatenate(y_real)
        y_pred = np.concatenate(y_pred)

        return classification_report(y_real,y_pred,target_names=label_names)