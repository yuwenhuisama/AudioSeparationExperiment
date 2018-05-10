'''
File: \estimate.py
Project: AudioSeparationExperiment
Created Date: Thursday May 10th 2018
Author: Huisama
-----
Last Modified: Thursday May 10th 2018 12:54:06 am
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import keras.backend as K
from keras.layers import Layer

class Estimate(Layer):
    def __init__(self, estms, estm_index, **kwargs):
        self.estms = estms
        self.estm_index = estm_index
        super(Estimate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Estimate, self).build(input_shape)

    def call(self, mixture):
        sum = K.sum(self.estms)
        mask = self.estms[self.estm_index] / (sum + K.epsilon())
        
        estimated = mixture * mask
        # print(mask.shape)
        return estimated

    def compute_output_shape(self, input_shape):
        return input_shape

