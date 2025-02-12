# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 8:18
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : VGG_19.py
# 本模型为VGG-19参考代码链接
import tensorflow as tf
 
 
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    return tf.nn.max_pool2d(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)
 
 
def dropout(x, keepPro, name=None):
    return tf.nn.dropout(x, keepPro, name)
 
 
def fcLayer(x, inputD, outputD, reluFlag, name):
    with tf.compat.v1.variable_scope(name) as scope:
        w = tf.compat.v1.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.compat.v1.get_variable("b", [outputD], dtype="float")
        out = tf.compat.v1.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out
 
 
def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding = "SAME"):
 
    channel = int(x.get_shape()[-1])
    with tf.compat.v1.variable_scope(name) as scope:
        w = tf.compat.v1.get_variable("w", shape=[kHeight, kWidth, channel, featureNum])
        b = tf.compat.v1.get_variable("b", shape=[featureNum])
        featureMap = tf.nn.conv2d(x, w, strides=[1, strideY, strideX, 1], padding=padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name)
 
 
class VGG19(object):
    def __init__(self, x, keepPro, classNum):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.begin_VGG_19()
 
    def begin_VGG_19(self):
        """build model"""
        conv1_1 = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1" )
        # print("conv1_1.shape:")
        # print(conv1_1.shape)
        # conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
        # print("conv1_2.shape:")
        # print(conv1_2.shape)
        pool1 = maxPoolLayer(conv1_1, 2, 2, 2, 2, "pool1")
        # print("pool1.shape:")
        # print(pool1.shape)
 
        conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
        # print("conv2_1.shape:")
        # print(conv2_1.shape)
        # conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
        # print("conv2_2.shape:")
        # print(conv2_2.shape)
        pool2 = maxPoolLayer(conv2_1, 2, 2, 2, 2, "pool2")
        # print("pool2.shape:")
        # print(pool2.shape)
 
        conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1")
        # print("conv3_1.shape:")
        # print(conv3_1.shape)
        # conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
        # # print("conv3_2.shape:")
        # # print(conv3_2.shape)
        # conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
        # # print("conv3_3.shape:")
        # # print(conv3_3.shape)
        # conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
        # print("conv3_4.shape:")
        # print(conv3_4.shape)
        pool3 = maxPoolLayer(conv3_1, 2, 2, 2, 2, "pool3")
        # print("pool3.shape:")
        # print(pool3.shape)
 
        conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
        # print("conv4_1.shape:")
        # print(conv4_1.shape)
        # conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
        # # print("conv4_2.shape:")
        # # print(conv4_2.shape)
        # conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
        # # print("conv4_3.shape:")
        # # print(conv4_3.shape)
        # conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
        # # print("conv4_4.shape:")
        # # print(conv4_4.shape)
        pool4 = maxPoolLayer(conv4_1, 2, 2, 2, 2, "pool4")
        # print("pool4.shape:")
        # print(pool4.shape)
 
        conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1")
        # print("conv5_1.shape:")
        # print(conv5_1.shape)
        # conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
        # # print("conv5_2.shape:")
        # # print(conv5_2.shape)
        # conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
        # # print("conv5_3.shape:")
        # # print(conv5_3.shape)
        # conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
        # print("conv5_4.shape:")
        # print(conv5_4.shape)
        pool5 = maxPoolLayer(conv5_1, 2, 2, 2, 2, "pool5")
        print('最后一层卷积层的形状是:', pool5.shape)
 
        fcIn = tf.reshape(pool5, [-1, 16*16*512])
        # print("fc6In.shape:")
        # print(fcIn.shape)
        fc6 = fcLayer(fcIn, 16*16*512, 4096, True, "fc6")
        # print("fc6.shape:")
        # print(fc6.shape)
        dropout1 = dropout(fc6, self.KEEPPRO)
 
        fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        # print("fc7.shape:")
        # print(fc7.shape)
        dropout2 = dropout(fc7, self.KEEPPRO)
 
        self.fc8 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")
 
 