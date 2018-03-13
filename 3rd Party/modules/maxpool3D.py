'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf
from module import Module

from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import nn_ops, gen_nn_ops


# 3D maxpooling 을 구현
# 기존의 코드에서 3Dmaxpooling을 추가로 구현
class MaxPool3D(Module):
    def __init__(self, pool_size=2, pool_stride=None, pad='SAME', name='maxpool3D'):
        self.name = name
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_kernel = [1, self.pool_size, self.pool_size, self.pool_size, 1]
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.stride_size = self.pool_size
            self.pool_stride = [1, self.stride_size, self.stride_size, self.stride_size, 1]
        self.pad = pad

    def forward(self, input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        self.in_N, self.in_h, self.in_c, self.in_w, self.in_depth = self.input_tensor.get_shape().as_list()

        # with tf.variable_scope(self.name):
        with tf.name_scope(self.name):
            self.activations = tf.nn.max_pool3d(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride,
                                              padding=self.pad, name=self.name)
            tf.summary.histogram('activations', self.activations)

        return self.activations

    def clean(self):
        self.activations = None
        self.R = None


    # LRP 상의 메모리 와 속도 문제를 해결하기 위해 기존의 코드를 수정 및 재구현
    # heatmapping.org의 tutorial에 나온 sudo코드 참조
    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        self.check_shape(R)

        in_N, in_h, in_w, in_c, in_depth = self.input_tensor.get_shape().as_list()
        Z = tf.nn.max_pool3d(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride, padding='SAME') + 1e-9
        S = self.R / Z
        C = gen_nn_ops._max_pool3d_grad(self.input_tensor, Z, S, ksize=self.pool_kernel, strides=self.pool_stride, padding='SAME')
        result = self.input_tensor*C
        return result


    def _alphabeta_lrp(self, R, alpha):
        '''
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        '''
        return self._simple_lrp(R)

    def check_shape(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape) != 4:
            activations_shape = self.activations.get_shape().as_list()
            if R_shape[0] == 1:
                self.batchsize_to_one()
            else:
                self.R = tf.reshape(self.R, activations_shape)
        N, self.Hout, self.Wout, self.Cout, NF = self.R.get_shape().as_list()
        if R_shape[0] == 1:
            self.batchsize_to_one()

    def batchsize_to_one(self):
        self.input_tensor = self.input_tensor[0, :, :, :]
        self.input_tensor = tf.expand_dims(self.input_tensor, 0)
        self.activations = self.activations[0, :, :, :]
        self.activations = tf.expand_dims(self.activations, 0)

        activations_shape = self.activations.get_shape().as_list()
        self.R = tf.reshape(self.R,
                            [activations_shape[0], activations_shape[1], activations_shape[2], activations_shape[3], activations_shape[4]])
        self.in_N = 1;
