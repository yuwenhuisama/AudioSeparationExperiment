'''
File: /network.py
Created Date: Thursday January 1st 1970
Author: huisama
-----
Last Modified: Monday May 7th 2018 1:40:18 am
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

        input = Input((40, 16, 8))
        bn = BatchNormalization()(input)
        conv = Conv2D(kernel_size=(1, 1), filters=16, padding='same', data_format='channels_first')(bn)
        conv = Conv2D(kernel_size=(2, 2), strides=(1, 1), filters=32, padding='same', data_format='channels_first')(conv)
        conv = Conv2D(kernel_size=(2, 2), strides=(1, 1), filters=64, padding='same', data_format='channels_first')(conv)
        conv = Conv2D(kernel_size=(2, 2), strides=(2, 2), filters=64, padding='same', data_format='channels_first')(conv)
        conv = Conv2D(kernel_size=(1, 1), strides=(1, 1), filters=40, padding='same', data_format='channels_first')(conv)

        flt = Flatten()(conv)
        # fcn = LeakyReLU()(Dense(units=1024)(flt))
        # fcn = LeakyReLU()(Dense(units=512)(fcn))
        # dp = Dropout(0.25)(fcn)
        bn = BatchNormalization()(flt)
        # fcn = Dense(units=256, activation='relu')(fcn)
        
        shared_encoder_model = Model(input, bn)
        return shared_encoder_model

    def get_separate_layer(self):
        seperate_inputs = Input((40, 32))

        merged_x = seperate_inputs
        self_attention = Attention(8, 32)([merged_x, merged_x, merged_x])
        # self_attention_flat = Flatten()(self_attention)

        lstm = LSTM(1024)(self_attention)
        # self_attention_flat = Flatten()(lstm)

        d1 = Dense(units=1024)
        fcn_decoded = BatchNormalization()(LeakyReLU()(d1(lstm)))
        d2 = Dense(units=2048)
        fcn_decoded = LeakyReLU()(d2(fcn_decoded))
        d3 = Dense(units=5120)
        seperate_model_acc = Model(inputs=seperate_inputs, outputs=d3(fcn_decoded), name='acc_output')
        seperate_model_voc = Model(inputs=seperate_inputs, outputs=d3(fcn_decoded), name='voc_output')

        return seperate_model_acc, seperate_model_voc

    def build_model(self):

        # encode layer
        shared_encoder_model = self.get_encoding_layer()

        # separate layer
        seperate_model_acc, seperate_model_voc = self.get_separate_layer()

        # orgin inputs
        inputs = Input((40, 16, 8))
        #inputs_reshape = [Reshape((-1,))(input) for input in inputs]
        encoded = shared_encoder_model(inputs)

        # output
        encoded = Reshape((40, -1))(encoded)
        output_acc = seperate_model_acc(encoded)
        output_voc = seperate_model_voc(encoded)

        # compile model
        # outputs_tmp = []
        # outputs_tmp.extend(output_acc)
        # outputs_tmp.extend(output_voc)

        model = Model(inputs=inputs, outputs=[output_acc, output_voc])
        
        # out = model.predict([np.random.random((1, 4, self.sfft_len, 1)) for i in range(10)])

        def costume_acc(y_true, y_pred):
            return K.mean(K.abs(y_true - y_pred))

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
            metrics={
                'acc_output': costume_acc,
                'voc_output': costume_acc
            }
        )

        self.model = model

        # rsp = Reshape((10, 4, self.sfft_len, 1))(fcn_decoded)
        # self._print_layer_shape(rsp)

    def train(self, database):
        for _ in range(50):
            mixture, vocals, accompaniment = database.generate_batch_data()
            y = [np.array(vocals), np.array(accompaniment)]
            self.model.fit(x=np.array(mixture),
                        y=y,
                        epochs=8,
                        batch_size=32)
            del mixture
            del vocals
            del accompaniment

            self.model.save('./model/model_%s.ckpt' % _)


    def predict(self):
        pass

'''
    Test
'''
database = Database('train')

nn = NetWork(SAMPLE_LEN, SFFT_BIN)
nn.build_model()

nn.train(database)
