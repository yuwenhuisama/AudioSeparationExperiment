'''
File: /network.py
Created Date: Thursday January 1st 1970
Author: huisama
-----
Last Modified: Thursday May 10th 2018 12:14:55 am
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, Layer, Reshape, Conv2DTranspose, Concatenate, LSTM, LeakyReLU, BatchNormalization, Dropout
from keras.optimizers import Adagrad, Adadelta
import keras.backend as K

import numpy as np

from resource import Database, DataOperation, BATCH_SIZE, FRN_BIN
from attention import Attention
from tfmask import TFMask

class NetWork:
    '''
        Neuron Network Class
    '''
    
    def get_encoding_layer(self):
        input = Input((BATCH_SIZE, 2, FRN_BIN))
        bn = BatchNormalization()(input)
        conv = BatchNormalization()(Conv2D(kernel_size=(1, 1), filters=16, padding='same', data_format='channels_first')(bn))
        conv = BatchNormalization()(Conv2D(kernel_size=(2, 2), strides=(1, 1), filters=32, padding='same', data_format='channels_first')(conv))
        conv = BatchNormalization()(Conv2D(kernel_size=(2, 2), strides=(1, 1), filters=64, padding='same', data_format='channels_first')(conv))
        conv = BatchNormalization()(Conv2D(kernel_size=(2, 2), strides=(2, 1), filters=64, padding='same', data_format='channels_first')(conv))
        conv = BatchNormalization()(Conv2D(kernel_size=(2, 2), strides=(2, 1), filters=128, padding='same', data_format='channels_first')(conv))
        conv = BatchNormalization()(Conv2D(kernel_size=(1, 1), strides=(2, 1), filters=64, padding='same', data_format='channels_first')(conv))

        flt = Flatten()(conv)
        fcn = LeakyReLU()(BatchNormalization()(Dense(units=2048)(flt)))
        # dp = Dropout(0.25)(fcn)
        # fcn = Dense(units=256, activation='relu')(fcn)
        
        shared_encoder_model = Model(input, fcn)
        return shared_encoder_model

    def get_estimate_layer(self):
        seperate_inputs = Input((128, 16))

        self_attention = Attention(8, 16)([seperate_inputs, seperate_inputs, seperate_inputs])
        
        print('self attention: ' + self_attention.shape)

        flat = Flatten()(self_attention)
        
        estm = LeakyReLU()(BatchNormalization()((Dense(1024)(flat))))
        estm = LeakyReLU()(BatchNormalization()((Dense(512)(estm))))
        output = LeakyReLU()(BatchNormalization()((Dense(2 * FRN_BIN)(estm))))

        estimate_model_acc = Model(inputs=seperate_inputs, outputs=output, name='acc_output')
        estimate_model_voc = Model(inputs=seperate_inputs, outputs=output, name='voc_output')

        return estimate_model_acc, estimate_model_voc

    def build_model(self):
        # encode layer
        shared_encoder_model = self.get_encoding_layer()

        # separate layer
        estm_acc, estm_voc = self.get_estimate_layer()

        # orgin inputs
        inputs = Input((BATCH_SIZE, 2, FRN_BIN))
        encoded = shared_encoder_model(inputs)

        # estm
        encoded = Reshape((128, -1))(encoded)
        output_acc = estm_acc(encoded)
        output_voc = estm_voc(encoded)

        # get tf masking
        voc_estimated = TFMask([output_acc, output_voc], 0)(inputs)
        acc_estimated = TFMask([output_acc, output_voc], 1)(inputs)

        model = Model(inputs=inputs, outputs=[voc_estimated, acc_estimated])
        
        def costume_acc(y_true, y_pred):
            return K.max(K.abs(y_true - y_pred))

        model.compile(
            optimizer=Adadelta(),
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

    def train(self, database):
        for _ in range(25):
            mixture, vocals, accompaniment = database.generate_batch_data()

            mixture = np.array(mixture)
            vocals = np.array(vocals)
            accompaniment = np.array(accompaniment)

            y = [vocals, accompaniment]
            self.model.fit(x=mixture,
                        y=y,
                        epochs=2,
                        batch_size=32)
            del mixture
            del vocals
            del accompaniment
            self.model.save('./model/model_%d.ckpt' % _)

    def train_with_model(self, database, model_path, continue_index):
        self.build_model()
        self.model.load_weights(model_path)

        database.batch_index = continue_index

        for _ in range(25 - continue_index // 2):
            mixture, vocals, accompaniment = database.generate_batch_data()

            mixture = np.array(mixture) * 100
            vocals = np.array(vocals) * 100
            accompaniment = np.array(accompaniment) * 100

            y = [vocals, accompaniment]
            self.model.fit(x=mixture,
                        y=y,
                        epochs=2,
                        batch_size=32)
            del mixture
            del vocals
            del accompaniment

        self.model.save('./model/model_%s.ckpt' % _)

    def predict_with_model(self, database, model_path):
        self.build_model()
        self.model.load_weights(model_path)

        data = database.generate_one_batch_data_for_test(0)  * 100

        result = self.model.predict(np.array(data))
        voices = result[0]
        accompaniments = result[1]

        voices = voices / 100
        accompaniments = accompaniments / 100

        # DataOperation.nn_output_to_wav(data, 'org.wav')
        DataOperation.nn_output_to_wav(voices, 'voice.wav')
        DataOperation.nn_output_to_wav(accompaniments, 'accompaniments.wav')

'''
    Test
'''
database = Database('train')

nn = NetWork()
nn.build_model()
nn.train(database)

# nn.train_with_model(database, './model/model_3.ckpt', 8)
# database2 = Database('sample')
# nn.predict_with_model(database2, './model/model_3.ckpt')
