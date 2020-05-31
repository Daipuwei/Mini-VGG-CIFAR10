# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 16:12
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : config.py
# @Software: PyCharm

import os

class config(object):

    default_dict = {
        "dataset_dir": os.path.abspath("./data/cifar10"),
        "checkpoints_dir": os.path.abspath("./checkpoints"),
        "logs_dir": os.path.abspath("./logs"),
        "result_dir": os.path.abspath("./result"),
        "config_dir": os.path.abspath("./config"),
        "input_image_shape": (32, 32, 3),
        "pre_model_path": None,
        "bacth_size": 16,
        "init_learning_rate": 0.01,
        "epoch": 50,
    }

    def __init__(self,**kwargs):
        """
        这是VGG16的初始化函数
        :param cfg: 参数配置类
        """
        # 初始化相关参数
        self.__dict__.update(self.default_dict)
        self.__dict__.update(kwargs)

        # 初始化相关目录
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        if not os.path.exists(self.config_dir):
            os.mkdir(self.config_dir)

    def save_logs(self, time):
        """
        这是保存模型训练相关参数的函数
        :param time: 时间
        :return:
        """
        # 创建本次训练相关目录
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, time)
        self.logs_dir = os.path.join(self.logs_dir, time)
        self.config_dir = os.path.join(self.config_dir, time)
        self.result_dir = os.path.join(self.result_dir, time)
        if not os.path.exists(self.config_dir):
            os.mkdir(self.config_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        config_txt_path = os.path.join(self.config_dir, "config.txt")
        with open(config_txt_path, 'a') as f:
            for key, value in self.__dict__.items():
                s = key + ": " + str(value) + "\n"
                f.write(s)