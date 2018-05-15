'''
File: /network-tf.py
Created Date: Thursday May 10th 2018
Author: huisama
-----
Last Modified: Tuesday May 15th 2018 2:48:19 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell

from resource import Database, DataOperation, BATCH_SIZE, FRN_BIN
from attention_tf import Attention
import math

class Network:

    def do_encoding(self, input):
        '''
            do_encoding(self, input)

            parameters:
            input: tensor of shape (batch_size, 2, frn_bin)

            return:
                tensor of shape(1024,)
        '''
        # Input (batch_size, 2, frn_bin)
        bn = tf.layers.batch_normalization(input)
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(64,1), filters=16, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(32,2), strides=(1,1), filters=32, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(16,2), strides=(2,1), filters=64, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(8,2), strides=(4,1), filters=128, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(4,1), strides=(8,1), filters=64, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(2,1), strides=(4,1), filters=64, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(1,1), strides=(2,1), filters=64, padding='same', data_format='channels_first'))
        conv = tf.layers.batch_normalization(tf.layers.conv2d(bn, kernel_size=(1,1), strides=(1,1), filters=32, padding='same', data_format='channels_first'))
        
        flt = tf.layers.flatten(conv)
        fcn = tf.layers.batch_normalization(tf.layers.dense(flt, units=2048))
        output = tf.nn.leaky_relu(fcn)
        return output

    def do_estimate(self, input, lstm_name):
        '''
            do_estimate(self, input)

            parameters:
            input: tensor of shape (32, 64)

            return:
                tensor of shape(2 * frn_bin)
        '''
        # input (32, 64)
        attention = Attention(input, input, input, 8, 128)

        # rnn
        with tf.variable_scope(lstm_name):
            rnn_layer = MultiRNNCell([GRUCell(4096) for _ in range(1)])
            output_rnn, _ = tf.nn.dynamic_rnn(rnn_layer, attention, dtype=tf.float32)            

        estm = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.dense(output_rnn, units=4096)))
        estm = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.dense(attention, units=2048)))
        output = tf.layers.batch_normalization(
                tf.layers.dense(estm, units=2 * FRN_BIN))

        output = tf.reshape(output, (-1, BATCH_SIZE, 2, FRN_BIN))

        return output

    def build_model(self):
        '''
            build_model(self)

            build model
        '''
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, 2, FRN_BIN))
        self.acc_src = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, 2, FRN_BIN))
        self.voc_src = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, 2, FRN_BIN))

        # encode
        encoded = self.do_encoding(self.input)

        # separate
        encoded = tf.reshape(encoded, shape=(-1, 32, 64))
        estm_acc = self.do_estimate(encoded, "gru_1")
        estm_voc = self.do_estimate(encoded, "gru_2")

        # tfmask
        tfmask_acc = estm_acc / (estm_acc + estm_voc + np.finfo(float).eps) * self.input
        tfmask_voc = estm_voc / (estm_acc + estm_voc + np.finfo(float).eps) * self.input

        result_acc = tf.reshape(tfmask_acc, (-1, BATCH_SIZE, 2, FRN_BIN))
        result_voc = tf.reshape(tfmask_voc, (-1, BATCH_SIZE, 2, FRN_BIN))

        self.acc = result_acc
        self.voc = result_voc

    def loss(self):
        gamma = 0.2
        return tf.reduce_mean(\
              tf.square(self.acc_src - self.acc) \
            + tf.square(self.voc_src - self.voc) \
            - gamma * tf.square(self.voc_src - self.acc) \
            - gamma * tf.square(self.acc_src - self.voc), name="loss")

    def train(self, database, continue_index=0):
        self.build_model()
        loss_fn = self.loss()
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer().minimize(loss_fn, global_step=global_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.load_state(sess, './model')

            for index in range(50-continue_index):
                mixture, vocals, accompaniment = database.generate_batch_data(continue_index+index)

                times = math.ceil(len(mixture) / 32)

                for j in range(8):
                    for i in range(times):
                        batch_mixture = mixture[i*32 : (i+1)*32 if (i+1)*32 < len(mixture) else len(mixture), :, :, :]
                        batch_voc = vocals[i*32 : (i+1)*32 if (i+1)*32 < len(vocals) else len(vocals), :, :, :]
                        batch_acc = accompaniment[i*32 : (i+1)*32 if (i+1)*32 < len(accompaniment) else len(accompaniment), :, :, :]

                        loss, _ = sess.run([loss_fn, optimizer], feed_dict={
                                                                self.acc_src: batch_acc,
                                                                self.voc_src: batch_voc,
                                                                self.input: batch_mixture
                                                            })

                        print("Turn %s, Epoch %s, Batch %s : loss %s: " % (index, i, j, loss))

                if index % 2 == 0:
                    self.save_state(sess, './model', continue_index+index)

    def save_state(self, sess, ckpt_path, global_step):
        print('Begin saving...')
        saver = tf.train.Saver()
        saver.save(sess, ckpt_path + '/checkpoint', global_step=global_step)
        print('End saving...')

    def load_state(self, sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

    def predict_with_model(self, database, model_path):
        self.build_model()
        data = database.generate_one_batch_data_for_test(0)
        # global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')  

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.load_state(sess, model_path)
            
            voc_arr = []
            acc_arr = []

            times = math.ceil(len(data) / 32)
            for i in range(times):
                batch_mixture = np.array(data[i*32 : (i+1)*32 if (i+1)*32 < len(data) else len(data), :, :, :])

                acc, voc = sess.run([self.acc, self.voc], feed_dict={self.input: batch_mixture})

                if (voc.shape[0] < BATCH_SIZE):
                    d = BATCH_SIZE - voc.shape[0]
                    padding = np.zeros((d, BATCH_SIZE, 2, FRN_BIN))
                    voc = np.concatenate((voc, padding))

                if (acc.shape[0] < BATCH_SIZE):
                    d = BATCH_SIZE - acc.shape[0]
                    padding = np.zeros((d, BATCH_SIZE, 2, FRN_BIN))
                    acc = np.concatenate((acc, padding))

                voc_arr.append(voc)
                acc_arr.append(acc)
            
            voices = np.array(voc_arr)
            accompaniments = np.array(acc_arr)

            voices = np.reshape(voices, (-1, BATCH_SIZE, 2, FRN_BIN))
            accompaniments = np.reshape(accompaniments, (-1, BATCH_SIZE, 2, FRN_BIN))

            # DataOperation.nn_output_to_wav(data, 'org.wav')
            DataOperation.nn_output_to_wav(voices, 'voice.wav')
            DataOperation.nn_output_to_wav(accompaniments, 'accompaniments.wav')

nn = Network()
# database = Database('train')
# nn.train(database, 0)

database2 = Database('train')
nn.predict_with_model(database2, './model')
