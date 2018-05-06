'''
File: /network.py
Created Date: Thursday January 1st 1970
Author: huisama
-----
Last Modified: Sunday May 6th 2018 11:06:48 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Layer, Reshape, Conv2DTranspose, Concatenate, LSTM, LeakyReLU, BatchNormalization, Dropout
from keras.optimizers import Adagrad, RMSprop
import keras.backend as K

import numpy as np

from resource import Database, DataOperation, SAMPLE_LEN, SFFT_BIN
from attention import Attention

class NetWork:
    '''
        Neuron Network Class
    '''
    def __init__(self, sample_len, sfft_len):
        self.sample_len = sample_len
        self.sfft_len = sfft_len

    def _print_layer_shape(self, layer):
        print(layer)

    def get_encoding_layer(self):
        # shared encoding layer
        # input = Input((4, self.sfft_len, 1))
        # conv = Conv2D(kernel_size=(2, 2), filters=16, padding='same')(input)
        # conv = Conv2D(kernel_size=(2, 2), strides=(2, 1), filters=32, padding='same')(conv)
        # conv = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=32, padding='same')(conv)
        # conv = Conv2D(kernel_size=(4, 4), strides=(4, 4), filters=64, padding='same')(conv)

        input = Input((2, self.sfft_len, 1))
        res = Reshape((64, 32, 1))(input)        
        bn = BatchNormalization()(res)
        conv = Conv2D(kernel_size=(1, 1), filters=16, padding='same')(bn)
        conv = Conv2D(kernel_size=(2, 2), strides=(1, 1), filters=32, padding='same')(conv)
        conv = Conv2D(kernel_size=(2, 2), strides=(2, 2), filters=32, padding='same')(conv)
        conv = Conv2D(kernel_size=(2, 2), strides=(2, 2), filters=64, padding='same')(conv)
        conv = Conv2D(kernel_size=(2, 2), strides=(2, 2), filters=64, padding='same')(conv)

        flt = Flatten()(conv)
        fcn = LeakyReLU()(Dense(units=1024)(flt))
        fcn = LeakyReLU()(Dense(units=512)(fcn))
        dp = Dropout(0.25)(fcn)
        bn = BatchNormalization()(dp)
        # fcn = Dense(units=256, activation='relu')(fcn)
        
        shared_encoder_model = Model(input, bn)
        return shared_encoder_model

    def get_separate_layer(self):
        seperate_inputs = [Input((512,)) for _ in range(self.sample_len)]

        aux_inputs = [Input((2048,)) for _ in range(self.sample_len)]

        merged_x = Concatenate(1)([BatchNormalization()(layer) for layer in seperate_inputs])
        merged_x = Reshape((-1, 512))(merged_x)
        self_attention = Attention(8, 256)([merged_x, merged_x, merged_x])
        # self_attention_flat = Flatten()(self_attention)

        lstm = LSTM(2048)(self_attention)
        # self_attention_flat = Flatten()(lstm)

        # self._print_layer_shape(self_attention) 

        merged_auxes = [Concatenate(1)
                ([lstm, BatchNormalization()(aux_inputs[i])]) for i in range(self.sample_len)]
        # self._print_layer_shape(merged_auxes[0])

        d1 = Dense(units=1024)
        fcn_decoded = [BatchNormalization()(LeakyReLU()(d1(layer))) for layer in merged_auxes]
        d2 = Dense(units=2048)
        fcn_decoded = [Dropout(0.25)(LeakyReLU()(d2(layer))) for layer in fcn_decoded]
        # bn = BatchNormalization()(fcn_decoded)
        # d3 = Dense(units=4096)
        # fcn_decoded = [d3(layer) for layer in fcn_decoded]
        
        inputs_tmp = []
        inputs_tmp.extend(seperate_inputs)
        inputs_tmp.extend(aux_inputs)
        seperate_model_acc = Model(inputs=inputs_tmp, outputs=fcn_decoded, name='acc_output')
        seperate_model_voc = Model(inputs=inputs_tmp, outputs=fcn_decoded, name='voc_output')

        return seperate_model_acc, seperate_model_voc

    def build_model(self):

        # encode layer
        shared_encoder_model = self.get_encoding_layer()

        # separate layer
        seperate_model_acc, seperate_model_voc = self.get_separate_layer()

        # orgin inputs
        inputs = [Input((2, self.sfft_len, 1)) for i in range(self.sample_len)]
        inputs_reshape = [Reshape((-1,))(input) for input in inputs]
        encoded = [shared_encoder_model(input) for input in inputs]

        # output
        inputs_tmp = []
        inputs_tmp.extend(encoded)
        inputs_tmp.extend(inputs_reshape)
        output_acc = seperate_model_acc(inputs_tmp)
        output_voc = seperate_model_voc(inputs_tmp)

        # compile model
        outputs_tmp = []
        outputs_tmp.extend(output_acc)
        outputs_tmp.extend(output_voc)

        model = Model(inputs=inputs, outputs=outputs_tmp)
        
        # out = model.predict([np.random.random((1, 4, self.sfft_len, 1)) for i in range(10)])

        model.compile(
            optimizer=Adagrad(),
            loss={
                'acc_output': 'cosine_proximity',
                'voc_output': 'cosine_proximity'
            },
            loss_weights={
                'acc_output': 1.0,
                'voc_output': 1.0
            },
        )

        self.model = model

        # rsp = Reshape((10, 4, self.sfft_len, 1))(fcn_decoded)
        # self._print_layer_shape(rsp)

    def train(self, database):
        for _ in range(50):
            mixture, vocals, accompaniment = database.generate_batch_data()
            vocals.extend(accompaniment)
            self.model.fit(x=mixture,
                        y=vocals,
                        epochs=4,
                        batch_size=32)
            del mixture
            del vocals
            del accompaniment

    def predict(self):
        pass

'''
    Test
'''
database = Database('sample')

nn = NetWork(SAMPLE_LEN, SFFT_BIN)
nn.build_model()

nn.train(database)
