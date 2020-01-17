"""The code was taken from carla-simulator/imitation-learning.

Code:
https://github.com/carla-simulator/imitation-learning/blob/master/agents/imitation/imitation_learning_network.py
This repository makes minor adjustment to this.
"""
from __future__ import print_function, unicode_literals

from future.builtins import object, range
import numpy as np
import tensorflow as tf

import constants as ilc

# from https://github.com/carla-simulator/imitation-learning/blob/62f93c2785a2452ca67eebf40de6bf33cea6cbce/agents/imitation/imitation_learning.py#L23  # noqa
DROPOUT_VEC_TRAIN = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
# DROPOUT_VEC_TRAIN = [1.0] * 19 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
DROPOUT_VEC_INFER = [1.0 for _ in DROPOUT_VEC_TRAIN]


def weight_xavi_init(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):
    def __init__(self, dropout, image_shape, is_training):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._is_training = is_training
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._conv_rate = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_{}'.format(self._count_conv))
        bias = bias_variable([output_size], name='B_c_{}'.format(self._count_conv))

        self._weights['W_conv{}'.format(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_{}'.format(self._count_conv)), bias,
                          name='add_{}'.format(self._count_conv))

        self._features['conv_block{}'.format(self._count_conv - 1)] = conv_res

        return conv_res

    def residual(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_{}'.format(self._count_conv))
        bias = bias_variable([output_size], name='B_c_{}'.format(self._count_conv))

        self._weights['W_conv{}'.format(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_{}'.format(self._count_conv)), bias,
                          name='add_{}'.format(self._count_conv))

        self._features['conv_block{}'.format(self._count_conv - 1)] = conv_res

        return conv_res

    def atrous_conv(self, x, kernel_size, rate, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_{}'.format(self._count_conv))
        bias = bias_variable([output_size], name='B_c_{}'.format(self._count_conv))

        self._weights['W_conv{}'.format(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_rate.append(rate)
        conv_res = tf.add(tf.nn.atrous_conv2d(x, weights, rate, padding=padding_in,
                                              name='conv2d_{}'.format(self._count_conv)), bias,
                          name='add_{}'.format(self._count_conv))

        self._features['conv_block{}'.format(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool{}'.format(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=self._is_training,
                                            updates_collections=None,
                                            scope='bn{}'.format(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu{}'.format(self._count_activations))

    def dropout(self, x):
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout{}'.format(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_{}'.format(self._count_fc))
        bias = bias_variable([output_size], name='B_f_{}'.format(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_{}'.format(self._count_fc))

    def atrous_conv_block(self, x, kernel_size, rate, output_size, padding_in='SAME'):
        with tf.name_scope("conv_block{}".format(self._count_conv)):
            x = self.atrous_conv(x, kernel_size, rate, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)
            return self.activation(x)

    # before conv activation is pre-activation
    # before Batch_norm + pre-activation is  full-pre-activation arichitecture
    def residual_block(self, x, kernel_size, stride, output_size, padding_in='SAME', change_dim=False):
        with tf.name_scope("residual_block{}".format(self._count_conv)):
            if change_dim:
                down_sample = self.max_pool(x)
                shortcut = self.conv_block(down_sample, 1, 1, output_size, padding_in='SAME')
                bn_1 = self.bn(x)
                relu_1 = self.activation(bn_1)
                conv_1 = self.residual(relu_1, kernel_size, 2, output_size,
                                       padding_in=padding_in)  ### stride 2 DownSampling

                bn_2 = self.bn(conv_1)
                relu_2 = self.activation(bn_2)
                conv_2 = self.residual(relu_2, kernel_size, stride, output_size, padding_in=padding_in)

                x_output = conv_2 + shortcut
            else:
                shortcut = x
                bn_1 = self.bn(x)
                relu_1 = self.activation(bn_1)
                conv_1 = self.residual(relu_1, kernel_size, stride, output_size,
                                       padding_in=padding_in)

                bn_2 = self.bn(conv_1)
                relu_2 = self.activation(bn_2)
                conv_2 = self.residual(relu_2, kernel_size, stride, output_size, padding_in=padding_in)

                x_output = conv_2 + shortcut

            return x_output

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        with tf.name_scope("conv_block{}".format(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)
            return self.activation(x)

    def fc_block(self, x, output_size):
        with tf.name_scope("fc{}".format(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block{}'.format(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features


def load_imitation_learning_network(input_image, input_data, mode):
    branches = []

    x = input_image
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout = DROPOUT_VEC_TRAIN
        print("dropout ", dropout)
        is_training = True
    else:
        dropout = DROPOUT_VEC_INFER
        is_training = False

    with tf.name_scope('Network'):  # for a nicer Tensorboard graph, use: `with tf.variable_scope('Network'):`
        network_manager = Network(dropout, tf.shape(x), is_training)
        # tf.image.per_image_standardization
        """conv1 - 44"""
        # whitening_image = tf.image.per_image_standardization(x)

        # 44 * 100
        # xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
        # res1 = network_manager.residual_block(xc, 3, 1, 32, padding_in='SAME', change_dim=False)
        #
        # res2 = network_manager.residual_block(res1, 3, 1, 32, padding_in='SAME', change_dim=False)
        # res3 = network_manager.residual_block(res2, 3, 1, 32, padding_in='SAME', change_dim=False)
        #
        # # 22 * 100
        # res4 = network_manager.residual_block(res3, 3, 1, 64, padding_in='SAME', change_dim=True)
        # res5 = network_manager.residual_block(res4, 3, 1, 64, padding_in='SAME', change_dim=False)
        #
        # # 11 * 100
        # res6 = network_manager.residual_block(res5, 3, 1, 128, padding_in='SAME', change_dim=True)
        # res7 = network_manager.residual_block(res6, 3, 1, 128, padding_in='SAME', change_dim=False)
        #
        # res7 = network_manager.conv_block(res7, 1, 1, 256, padding_in='SAME')
        #
        # res8 = network_manager.residual_block(res7, 3, 1, 256, padding_in='SAME', change_dim=False)
        # res9 = network_manager.residual_block(res8, 3, 1, 256, padding_in='SAME', change_dim=False)

        # xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='SAME')
        #
        # xc5 = network_manager.conv_block(x, 5, 2, 32, padding_in='SAME')
        #
        # xc7 = network_manager.conv_block(x, 7, 2, 32, padding_in='SAME')
        #
        # xc = tf.concat([xc, xc5, xc7], -1)
        #
        # """conv2"""
        # xc5 = network_manager.conv_block(xc, 5, 2, 64, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='SAME')
        #
        # xc = tf.concat([xc, xc5], -1)
        #
        # """conv3"""
        # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='SAME')
        #
        # """conv4"""
        # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')

        # xc = network_manager.conv_block(xx, 3, 1, 256, padding_in='SAME')
        """conv1"""  # kernel sz, stride, num feature maps
        # xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='SAME')
        #
        # xc5 = network_manager.conv_block(x, 5, 2, 32, padding_in='SAME')
        #
        # xc7 = network_manager.conv_block(x, 7, 2, 32, padding_in='SAME')
        #
        # xc = tf.concat([xc, xc5, xc7], -1)
        #
        # """conv2"""
        # xc5 = network_manager.conv_block(xc, 5, 2, 64, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='SAME')
        #
        # xc = tf.concat([xc, xc5], -1)
        #
        # """conv3"""
        # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='SAME')
        #
        # """conv4"""
        # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
        # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')

        """mp3 (default values)"""
        """conv1"""  # kernel sz, stride, num feature maps
        xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')

        """conv2"""
        xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')

        """conv3"""
        xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')

        """conv4"""
        xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
        xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
        """mp3 (default values)"""
        # -----------------------------------------------------------------------# 88*200
        # xc1_o = network_manager.conv_block(x, 5, 2, 32, padding_in='SAME')
        # xc1_o = network_manager.conv_block(x, 1, 1, 32, padding_in='SAME')
        # xc1 = network_manager.conv_block(xc1_o, 1, 1, 16, padding_in='SAME')
        # xc1 = network_manager.conv_block(xc1, 3, 1, 16, padding_in='SAME')
        # xc1 = network_manager.conv_block(xc1, 1, 1, 32, padding_in='SAME')
        #
        # xc2_o = tf.concat([xc1_o, xc1], -1)    #64 channel
        #
        # xc2 = network_manager.conv_block(xc2_o, 1, 1, 32, padding_in='SAME')
        # xc2 = network_manager.conv_block(xc2, 3, 1, 32, padding_in='SAME')
        # xc2 = network_manager.conv_block(xc2, 1, 1, 64, padding_in='SAME')
        #
        # xc3 = tf.concat([xc2, xc2_o], -1)  # 128 channel
        # xc3_o = network_manager.max_pool(xc3)
        # print("1 ", xc2.shape, xc2_o.shape, xc3.shape)
        # #-----------------------------------------------------------------------# 44*100
        #
        # xc3 = network_manager.conv_block(xc3_o, 1, 1, 64, padding_in='SAME')
        # xc3 = network_manager.conv_block(xc3, 3, 1, 64, padding_in='SAME')
        # xc3 = network_manager.conv_block(xc3, 1, 1, 128, padding_in='SAME')
        #
        # xc4_o = tf.concat([xc3_o, xc3], -1)  # 256 channel
        #
        # xc4 = network_manager.conv_block(xc4_o, 1, 1, 128, padding_in='SAME')
        # xc4 = network_manager.conv_block(xc4, 3, 1, 128, padding_in='SAME')
        # xc4 = network_manager.conv_block(xc4, 1, 1, 256, padding_in='SAME')
        #
        # xc5 = tf.concat([xc4_o, xc4], -1)  # 512 channel
        # xc5_o = network_manager.max_pool(xc5)
        # print("2 ", xc4.shape, xc4_o.shape, xc5.shape)
        # # -----------------------------------------------------------------------# 22*100
        #
        # xc5 = network_manager.conv_block(xc5_o, 1, 1, 256, padding_in='SAME')
        # xc5 = network_manager.conv_block(xc5, 3, 1, 256, padding_in='SAME')
        # xc5 = network_manager.conv_block(xc5, 1, 1, 512, padding_in='SAME')
        #
        # xc6_o = tf.concat([xc5_o, xc5], -1)  # 1024 channel
        #
        # xc6 = network_manager.conv_block(xc6_o, 1, 1, 512, padding_in='SAME')
        # xc6 = network_manager.conv_block(xc6, 3, 1, 512, padding_in='SAME')
        # xc6 = network_manager.conv_block(xc6, 1, 1, 1024, padding_in='SAME')
        #
        # xc7 = tf.concat([xc6_o, xc6], -1)  # 2048 channel

        # print("3 ", xc6.shape, xc6_o.shape, xc7.shape)
        # -----------------------------------------------------------------------#
        """ reshape """
        # multi_scale = tf.concat([xc, xc1], 1)
        x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')

        """ fc1 """
        x = network_manager.fc_block(x, 512)
        # x = network_manager.fc_block(x, 1024)
        """ fc2 """
        x = network_manager.fc_block(x, 512)
        # x = network_manager.fc_block(x, 1024)
        """Process Control"""

        """ Speed (measurements)"""
        with tf.name_scope("Speed"):
            speed = input_data[1]  # get the speed from input data
            speed = network_manager.fc_block(speed, 128)
            speed = network_manager.fc_block(speed, 128)

        """ Joint sensory """
        j = tf.concat([x, speed], 1)
        j = network_manager.fc_block(j, 512)

        """Start BRANCHING"""
        branch_config = [
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_STEER, ilc.TGT_GAS, ilc.TGT_BRAKE],
            [ilc.TGT_SPEED],
        ]
        for i in range(0, len(branch_config)):
            with tf.name_scope("Branch_{}".format(i)):
                if branch_config[i][0] == ilc.TGT_SPEED:
                    # we only use the image as input to speed prediction
                    branch_output = network_manager.fc_block(x, 256)
                    branch_output = network_manager.fc_block(branch_output, 256)
                else:
                    branch_output = network_manager.fc_block(j, 256)
                    branch_output = network_manager.fc_block(branch_output, 256)

                branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        return branches
        # branch_config = [["Steer", "Gas", "Brake"],
        #                  ["Steer", "Gas", "Brake"],
        #                  ["Steer", "Gas", "Brake"],
        #                  ["Steer", "Gas", "Brake"],
        #                  ["Speed"]]
        #
        # for i in range(0, len(branch_config)):
        #     with tf.name_scope("Branch_" + str(i)):
        #         if branch_config[i][0] == "Speed":
        #             # we only use the image as input to speed prediction
        #             branch_output = network_manager.fc_block(x, 256)
        #             branch_output = network_manager.fc_block(branch_output, 256)
        #         else:
        #             branch_output = network_manager.fc_block(j, 256)
        #             branch_output = network_manager.fc_block(branch_output, 256)
        #
        #         branches.append(network_manager.fc(branch_output, len(branch_config[i])))
        #     # print(branch_output)
        # return branches
