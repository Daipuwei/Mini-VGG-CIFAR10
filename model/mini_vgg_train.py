# -*- coding: utf-8 -*-
# @Time    : 2020/5/24 17:11
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : mini_vgg_train.py
# @Software: PyCharm

import os
import datetime
import numpy as np
from matplotlib import pyplot as plt

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization

from keras import Model
from keras.optimizers import Adam
from keras import backend as K

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

class mini_VGG(object):

    def __init__(self,cfg):
        """
        这是VGG16的初始化函数
        :param cfg: 参数配置类
        """
        # 初始化相关参数
        self.cfg = cfg

        # 搭建MiNi-VGG，并编译模型
        self.build_model()
        """
        self.model.compile(optimizer=SGD(lr=self.init_learning_rate, momentum=0.9,
                                         nesterov=True,decay= 0.01 / self.epoch),
                           loss=["categorical_crossentropy"],metrics=["acc"])
        """
        self.model.compile(optimizer=Adam(lr=self.cfg.init_learning_rate),
                           loss=["categorical_crossentropy"], metrics=["acc"])
        if self.cfg.pre_model_path is not None:
            self.model.load_weights(self.cfg.pre_model_path,by_name=True,skip_mismatch=True)
            print("loads model from: ",self.cfg.pre_model_path)

    def build_model(self):
        """
        这是Mini-VGG网络的搭建函数
        :return:
        """
        # 初始化网络输入
        self.image_input = Input(shape=self.cfg.input_image_shape,name="image_input")

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

    def train(self,train_datagen,val_datagen,train_iter_num,val_iter_num,init_epoch=0):
        """
        这是VGG16的训练函数
        :param train_datagen: 训练数据集生成器
        :param val_datagen: 验证数据集生成器
        :param train_iter_num: 一个epoch训练迭代次数
        :param val_iter_num: 一个epoch验证迭代次数
        :param init_epoch: 初始周期数
        """
        # 初始化相关文件目录路径,并保存到日志文件
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.cfg.save_logs(time)

        # 初始化回调函数
        tensorboard = TensorBoard(self.cfg.logs_dir,)
        early_stop = EarlyStopping(monitor='val_loss',min_delta=1e-6,verbose=1,patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patience=2)
        checkpint1 = ModelCheckpoint(filepath=os.path.join(self.cfg.checkpoints_dir,
                                                          'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-'
                                                          'acc{acc:.3f}-val_acc{val_acc:.3f}.h5'),
                                    monitor='val_loss', save_best_only=True,verbose=1)
        checkpint2 = ModelCheckpoint(filepath=os.path.join(self.cfg.checkpoints_dir,
                                                          'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-'
                                                          'acc{acc:.3f}-val_acc{val_acc:.3f}.h5'),
                                    monitor='val_acc', save_best_only=True, verbose=1)

        # 训练模型--第一阶段
        history1 = self.model.fit_generator(train_datagen,steps_per_epoch=train_iter_num,
                                 validation_data=val_datagen,validation_steps=val_iter_num,verbose=1,
                                 initial_epoch=init_epoch,epochs=self.cfg.epoch,
                                 callbacks=[tensorboard,checkpint1,checkpint2,early_stop,reduce_lr])
        self.model.save(os.path.join(self.cfg.checkpoints_dir,"stage1-trained-model.h5"))

        # 冻结除最后预测分类的全连接层之外的所有层参数
        for i in range(len(self.model.layers)-1):
            self.model.layers[i].trainable = False

        # 重新设置学习率
        K.set_value(self.model.optimizer.lr,self.cfg.init_learning_rate // 100)

        # 初始化回调函数
        checkpint1 = ModelCheckpoint(filepath=os.path.join(self.cfg.checkpoints_dir,
                                                           'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-'
                                                           'acc{acc:.3f}-val_acc{val_acc:.3f}.h5'),
                                     monitor='val_loss', save_best_only=True, verbose=1)
        checkpint2 = ModelCheckpoint(filepath=os.path.join(self.cfg.checkpoints_dir,
                                                           'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-'
                                                           'acc{acc:.3f}-val_acc{val_acc:.3f}.h5'),
                                     monitor='val_acc', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=2)
        history2 = self.model.fit_generator(train_datagen, steps_per_epoch=train_iter_num,
                                 validation_data=val_datagen, validation_steps=val_iter_num, verbose=1,
                                 initial_epoch=self.cfg.epoch, epochs=self.cfg.epoch*2,
                                 callbacks=[tensorboard, checkpint1,checkpint2,early_stop,reduce_lr])
        self.model.save(os.path.join(self.cfg.checkpoints_dir, "stage2-trained-model.h5"))

        # 绘制训练与验证损失走势图
        loss = np.concatenate([history1.history["loss"],history2.history["loss"]])
        val_loss = np.concatenate([history1.history["val_loss"], history2.history["val_loss"]])
        plt.plot(np.arange(0, len(loss)), loss, label="train_loss")
        plt.plot(np.arange(0, len(val_loss)), val_loss, label="val_loss")
        plt.title("Loss on CIFAR-10")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.result_dir,"loss.png"))
        plt.close()

        # 绘制训练与验证精度走势图
        acc = np.concatenate([history1.history["acc"], history2.history["acc"]])
        val_acc = np.concatenate([history1.history["val_acc"], history2.history["val_acc"]])
        plt.plot(np.arange(0, len(acc)), acc, label="train_acc")
        plt.plot(np.arange(0, len(val_acc)), val_acc, label="val_acc")
        plt.title("Accuracy on CIFAR-10")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.result_dir, "accuracy.png"))
        plt.close()