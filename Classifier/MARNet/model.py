# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:22:03 2023

@author: dell
"""
from keras.layers import ReLU, Dropout, GRU
from keras.layers import Conv1D, Dense, AveragePooling1D, Layer, Input, MaxPooling1D, BatchNormalization, Add,  GRU
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns


class MyMultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = kernel_initializer
        super(MyMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo', shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        attentions = []
        for i in range(self.W.shape[0]):  # 多个头循环计算
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            attentions.append(e)  # 存储当前头的注意力权重
            o = K.batch_dot(e, v)
            if i == 0:
                outputs = o
            else:
                outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z, attentions

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def residual_block(x, kernel_size=1):
    y = BatchNormalization()(x)
    y = ReLU()(y)
    y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same')(y)
    projection = GRU(64)(x)
    projection = K.expand_dims(projection, axis=2)
    y = Add()([projection, y])
    return y


    
    

