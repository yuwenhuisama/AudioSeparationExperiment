'''
File: \tfmask.py
Project: AudioSeparationExperiment
Created Date: Wednesday May 9th 2018
Author: Huisama
-----
Last Modified: Thursday May 10th 2018 11:31:42 am
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import keras.backend as K
from keras.layers import Layer, RepeatVector
from resource import BATCH_SIZE

class TFMask(Layer):
    def __init__(self, **kwargs):
        super(TFMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TFMask, self).build(input_shape)

    def call(self, x):
        sum = x[1] + x[2] + K.epsilon()
        # print(sum.shape)
        mask = x[1] / sum

        mask = RepeatVector(BATCH_SIZE)(mask)

        estimated = mask * x[0]
        return estimated

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
