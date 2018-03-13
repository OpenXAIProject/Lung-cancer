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

import pdb
from math import ceil

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import nn_ops, gen_nn_ops
import activations
import variables
from module import Module


# convolution을 3D환경에 맞게 수정하기 위해 3Dconvolution의 모듈을 만듬

class Convolution3D(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, output_depth, batch_size=None, input_dim = None, input_depth=None, kernel_size=5, stride_size=2, act = 'relu', phase=True, pad = 'SAME', weights_init= tf.truncated_normal_initializer(stddev=0.01), bias_init= tf.constant_initializer(0.0), final = False,name="conv3d"):
        self.name = name
        #self.input_tensor = input_tensor
        Module.__init__(self)


        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_depth = input_depth


        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.kernel_channels = kernel_size
        self.stride_size = stride_size
        self.act = act
        self.pad = pad

        self.weights_init = weights_init
        self.bias_init = bias_init

        self.momentum = 0.9
        self.epsilon = 0.001
        self.training = phase
        self.final_layer = final

    def check_input_shape(self):
        inp_shape = self.input_tensor.get_shape().as_list()
        try:
            if len(inp_shape)!=5:
                mod_shape = [self.batch_size, self.input_dim, self.input_dim,self.input_dim,self.input_depth]
                self.input_tensor = tf.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError('Expected dimension of input tensor: 4')

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #pdb.set_trace()
        self.check_input_shape()
        self.in_N, self.in_h, self.in_w, self.in_c, self.in_depth = self.input_tensor.get_shape().as_list()

        # init weights
        self.weights_shape = [self.kernel_size, self.kernel_size, self.kernel_channels, self.in_depth, self.output_depth]
        self.strides = [1, self.stride_size, self.stride_size, self.stride_size, 1]
        with tf.variable_scope(self.name):
            self.weights = variables.weights(self.weights_shape, initializer=self.weights_init, name=self.name)
            self.biases = variables.biases(self.output_depth, initializer=self.bias_init, name=self.name)

        with tf.name_scope(self.name):
            # convolutional layer뒤에 batch_normalization이 붙도록 코드를 수정
            if self.final_layer == False:
                conv = tf.nn.conv3d(self.input_tensor, self.weights, strides = self.strides, padding=self.pad)
                bn = tf.contrib.layers.batch_norm(conv, decay=self.momentum,
                                            updates_collections=None, epsilon=self.epsilon,
                                                          scale=False, is_training=self.training)

                if isinstance(self.act, str):
                    self.activations = activations.apply(bn, self.act)
                elif hasattr(self.act, '__call__'):
                    self.activations = self.act(bn)
                print('not final')
                #
                # if self.keep_prob<1.0:
                #     self.activations = tf.nn.dropout(self.activations, keep_prob=self.keep_prob)
                #
                tf.summary.histogram('activations', self.activations)
                tf.summary.histogram('weights', self.weights)
                tf.summary.histogram('biases', self.biases)
            else:
                conv = tf.nn.conv3d(self.input_tensor, self.weights, strides=self.strides, padding=self.pad)
                self.activations = conv
                print('final')
        return self.activations

    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        
        self.check_shape(R)

        image_patches = self.extract_patches()
        Z = self.compute_z(image_patches)
        Zs = self.compute_zs(Z)
        result = self.compute_result(Z,Zs)
        return self.restitch_image(result)

    

    def check_shape(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=5:
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
    def clean(self):
        self.activations = None
        self.R = None
    
    def __simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        #out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()


        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr/2
            p_bottom = pr-(pr/2)
            p_left = pc/2
            p_right = pc-(pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[p_top,p_bottom],[p_left, p_right],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0),0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                term2 =  tf.expand_dims(input_slice, -1)
                #pdb.set_trace()
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                #Zs = t1 + t2
                Zs = t1
                stabilizer = 1e-8*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
                Zs += stabilizer
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3), 4)
                
                #pdb.set_trace()
                #pad each result to the dimension of the out
                pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_top = i*hstride
                pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_left = j*wstride
                result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")
                # print(i,j)
                # print(i*hstride, i*hstride+hf , j*wstride, j*wstride+wf)
                # print(pad_top, pad_bottom,pad_left, pad_right)
                Rx+= result
        #pdb.set_trace()
        total_time = time.time() - start_time
        print(total_time)
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx

    # 메모리문제를 해결하기 위해 기존의 alphabeta lrp에서 코드부분을 수정함
    # heatmapping.org에 있는 tutorial에 있는 알고리즘에 따라 코드를 구현

    def _alphabeta_lrp(self, R, alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta =alpha -1
        self.R = R
        self.check_shape(R)
        _, hstride, wstride, cstride, _ = self.strides

        tmp_weight = tf.maximum(1e-9, self.weights)
        tmp2_weight = tf.minimum(-1e-9, self.weights)
        X = self.input_tensor+1e-9
        Za = tf.nn.conv3d(X, tmp_weight, strides=self.strides, padding='SAME')
        Sa = alpha*self.R/Za
        Zb = tf.nn.conv3d(X, tmp2_weight, strides=self.strides, padding='SAME')
        Sb = -beta * self.R / Zb
        result = X*(nn_ops.conv3d_backprop_input_v2(tf.shape(self.input_tensor), tmp_weight, Sa, strides=self.strides, padding='SAME')
        +nn_ops.conv3d_backprop_input_v2(tf.shape(self.input_tensor), tmp2_weight, Sb, strides=self.strides, padding='SAME'))
        return result