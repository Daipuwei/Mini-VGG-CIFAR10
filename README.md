# Mini-VGG-CIFAR10
这是利用Mini-VGG实现CIFAR10数据集训练与评估的函数

# 环境

 - Python 3.6
 - Tensorflow 1.14.0
 - Keras 2.1.5
 - OpenCV 3.4.5

# Train

 1. 在[CIFAR官网](http://www.cs.toronto.edu/~kriz/cifar.html)下载CIFAR10数据集，并放置在`./data`
 2. 运行代码`python cifar_preprocess.py`，将官网CIFAR10训练集格式转换为Keras官方ImageDataGenerator所支持的格式。
 3. 运行`python train_mini_vgg.py`代码进行训练Mini-VGG模型

# Evaluate
运行代码`python eval_on_dataset.py`即可对在指定数据集上评估Mini-VGG性能。

# CIFAR10实验结果
*训练过程中损失和精度走势图如下：*

![在这里插入图片描述](https://img-blog.csdnimg.cn/202005311624413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/202005311624414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)

*训练结束之后，Mini-VGG在CIFAR10的验证集上的评估结果如下：*

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200531162604295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwMDkxOTQ1,size_16,color_FFFFFF,t_70#pic_center)
