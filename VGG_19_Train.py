# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 16:07
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : VGG_19_Train.py
# 定义一些模型中所需要的参数
from VGG_19 import VGG19
import tensorflow as tf
import os
import cv2
import numpy as np
from keras.utils import to_categorical
 
batch_size = 64
img_high = 100
img_width = 100
Channel = 3
label = 9
 
# 定义输入图像的占位符
inputs = tf.placeholder(tf.float32, [batch_size, img_high, img_width, Channel], name='inputs')
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, label], name='label')
keep_prob = tf.placeholder("float")
is_train = tf.placeholder(tf.bool)
 
model = VGG19(inputs, keep_prob, label)
score = model.fc8
softmax_result = tf.nn.softmax(score)
 
# 定义损失函数 以及相对应的优化器
cross_entropy = -tf.reduce_sum(y*tf.log(softmax_result))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 
# 显示最后预测的结果
correct_prediction = tf.equal(tf.argmax(softmax_result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
 
# 现在的我只需要加载图像和对应的label即可 不需要加载text中的内容
def load_satetile_image(batch_size=128, dataset='train'):
    img_list = []
    label_list = []
    dir_counter = 0
 
    if dataset == 'train':
        path = '../Dataset/baidu/train_image/train'
 
        # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            for dir_image in os.listdir(child_path):
                img = cv2.imread(os.path.join(child_path, dir_image))
                img = img / 255.0
                img_list.append(img)
                label_list.append(dir_counter)
 
            dir_counter += 1
    else:
        path = '../Dataset/baidu/valid_image/valid'
 
        # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            for dir_image in os.listdir(child_path):
                img = cv2.imread(os.path.join(child_path, dir_image))
                img = img / 255.0
                img_list.append(img)
                label_list.append(dir_counter)
 
            dir_counter += 1
 
    # 返回的img_list转成了 np.array的格式
    X_train = np.array(img_list)
    Y_train = to_categorical(label_list, 9)
    # print('to_categorical之后Y_train的类型和形状:', type(Y_train), Y_train.shape)
 
    # 加载数据的时候 重新排序
    data_index = np.arange(X_train.shape[0])
    np.random.shuffle(data_index)
    data_index = data_index[:batch_size]
    x_batch = X_train[data_index, :, :, :]
    y_batch = Y_train[data_index, :]
 
    return x_batch, y_batch
 
 
# 开始feed 数据并且训练数据
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500000//batch_size):
        # 加载训练集和验证集
        img, img_label = load_satetile_image(batch_size, dataset='train')
        img_valid, img_valid_label = load_satetile_image(batch_size, dataset='vaild')
        # print('使用 mnist.train.next_batch加载的数据集形状', img.shape, type(img))
 
        # print('模型使用的是dropout的模型')
        dropout_rate = 0.5
        # print('经过 tf.reshape之后数据的形状以及类型是:', img.shape, type(img))
        if i % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={inputs: img, y: img_label, keep_prob: dropout_rate})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={inputs: img, y: img_label, keep_prob: dropout_rate})
 
        # 输出验证集上的结果
        if i % 50 == 0:
            dropout_rate = 1
            valid_socre = accuracy.eval(feed_dict={inputs: img_valid, y: img_valid_label, keep_prob: dropout_rate})
            print("step %d, valid accuracy %g" % (i, valid_socre))