'''
File: /network.py
Created Date: Thursday January 1st 1970
Author: huisama
-----
Last Modified: Sat May 05 2018
Modified By: huisama
-----
Copyright (c) 2018 Hui
'''

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Layer, Reshape, Conv2DTranspose, Concatenate
from keras.optimizers import Adagrad
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

    def build_model(self):
        # shared encoding layer
        input = Input((4, self.sfft_len, 1))
        conv = Conv2D(kernel_size=(2, 2), filters=16, padding='same')(input)
        conv = Conv2D(kernel_size=(2, 2), strides=(2, 1), filters=32, padding='same')(conv)
        conv = Conv2D(kernel_size=(4, 4), strides=(2, 2), filters=32, padding='same')(conv)
        conv = Conv2D(kernel_size=(4, 4), strides=(4, 4), filters=64, padding='same')(conv)

        flt = Flatten()(conv)
        fcn = Dense(units=1024)(flt)
        fcn = Dense(units=512)(fcn)
        fcn = Dense(units=256)(fcn)
        
        shared_encoder_model = Model(input, fcn)

        # separate layer
        seperate_inputs = [Input((256,)) for _ in range(self.sample_len)]
        aux_inputs = [Input((4096,)) for _ in range(self.sample_len)]

        merged_x = Concatenate(1)(seperate_inputs)
        merged_x = Reshape((-1, 256))(merged_x)
        self_attention = Attention(8, 256)([merged_x, merged_x, merged_x])
        self_attention_flat = Flatten()(self_attention)
        self._print_layer_shape(self_attention) 

        merged_auxes = [Concatenate(1)([self_attention_flat, aux_inputs[i]]) for i in range(self.sample_len)]
        self._print_layer_shape(merged_auxes[0])

        d1 = Dense(units=1024)
        fcn_decoded = [d1(layer) for layer in merged_auxes]
        d2 = Dense(units=2048)
        fcn_decoded = [d2(layer) for layer in fcn_decoded]
        d3 = Dense(units=4096)
        fcn_decoded = [d3(fcn_decoded[i]) for i in range(len(fcn_decoded))]
        
        inputs_tmp = []
        inputs_tmp.extend(seperate_inputs)
        inputs_tmp.extend(aux_inputs)
        seperate_model_acc = Model(inputs=inputs_tmp, outputs=fcn_decoded, name='acc_output')
        seperate_model_voc = Model(inputs=inputs_tmp, outputs=fcn_decoded, name='voc_output')        

        # orgin inputs
        inputs = [Input((4, self.sfft_len, 1), name='org_input_%s' % i) for i in range(self.sample_len)]
        inputs_reshape = [Reshape((-1,))(input) for input in inputs]
        # inputs_reshape = inputs
        print(inputs_reshape)
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

        # for o in range(len(outputs_tmp)):
        #     outputs_tmp.name = 'org_output_%s' % o

        model = Model(inputs=inputs, outputs=outputs_tmp)
        
        # out = model.predict([np.random.random((1, 4, self.sfft_len, 1)) for i in range(10)])

        model.compile(
            optimizer=Adagrad(),
            loss={
                'acc_output': 'mean_squared_error',
                'voc_output': 'mean_squared_error'
            },
            loss_weights={
                'acc_output': 1.0,
                'voc_output': 1.0
            },
            metrics= {
                'acc_output': ['accuracy'],
                'voc_output': ['accuracy']
            }
        )

        self.model = model

        # rsp = Reshape((10, 4, self.sfft_len, 1))(fcn_decoded)
        # self._print_layer_shape(rsp)

    def train(self, database):
        mixture, vocal, accompaniment = database.generate_batch_data()
        
        x = {}
        for i in range(SAMPLE_LEN):
            x['org_input_%s' % i] = mixture[i]

        vocal.extend(accompaniment)

        self.model.fit(x=x,
                        y=vocal,
                        epochs=64,
                        batch_size=32)

    def predict(self):
        pass

'''
    Test
'''
database = Database('sample')

nn = NetWork(SAMPLE_LEN, SFFT_BIN)
nn.build_model()

nn.train(database)
