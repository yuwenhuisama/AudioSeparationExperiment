'''
File: /network-tf.py
Created Date: Thursday May 10th 2018
Author: huisama
-----
Last Modified: Thu May 17 2018
Modified By: huisama
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

import gc

class Network:

    def do_encoding(self, input):
        '''
            do_encoding(self, input)

            parameters:
            input: tensor of shape (-1, batch_size, frn_bin)

            return:
                tensor of shape(-1, batch_size, 2048)
        '''
        # Input (none, databatch_size, batch_size, frn_bin)
        with tf.variable_scope('encoding_layer', reuse=tf.AUTO_REUSE):
            bn = tf.layers.batch_normalization(input)
            # print(bn.shape)
            flt = tf.layers.flatten(bn)
            fcn = tf.layers.batch_normalization(tf.layers.dense(flt, units=2048))
            fcn = tf.nn.leaky_relu(fcn)
            fcn = tf.layers.batch_normalization(tf.layers.dense(bn, units=1024))
            output = tf.nn.leaky_relu(fcn)
        return output

    def do_estimate(self, input, lstm_name, scope_name):
        '''
            do_estimate(self, input)

            parameters:
            input: tensor of shape (-1, batch_size, 1024)

            return:
                tensor of shape(-1, batch_size, frn_bin)
        '''
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            # input (32, 64)
            attention = Attention(input, input, input, 8, 16)
            # attention = tf.reshape(attention, (tf.shape(input)[0],))

            # rnn
            with tf.variable_scope(lstm_name):
                rnn_layer = MultiRNNCell([GRUCell(4096) for _ in range(2)])
                output_rnn, _ = tf.nn.dynamic_rnn(rnn_layer, attention, dtype=tf.float32)

            estm = tf.nn.leaky_relu(
                tf.layers.batch_normalization(
                    tf.layers.dense(output_rnn, units=4096)))
            estm = tf.nn.leaky_relu(
                tf.layers.batch_normalization(
                    tf.layers.dense(attention, units=1024)))
            output = tf.layers.batch_normalization(
                    tf.layers.dense(estm, units=FRN_BIN))

            output = tf.reshape(output, (-1, BATCH_SIZE, FRN_BIN))

        return output

    def tf_mask(self, source, target):
        return source / (source + target + np.finfo(float).eps)

    def build_model(self):
        '''
            build_model(self)

            build model
        '''
        self.input_left = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, FRN_BIN))
        self.input_right = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, FRN_BIN))
        self.acc_src_left = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, FRN_BIN))
        self.acc_src_right = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, FRN_BIN))        
        self.voc_src_left = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, FRN_BIN))
        self.voc_src_right = tf.placeholder(dtype=tf.float32, shape=(None, BATCH_SIZE, FRN_BIN))

        # encode
        encoded_left = self.do_encoding(self.input_left)
        encoded_right = self.do_encoding(self.input_right)

        # separate
        # encoded_left = tf.reshape(encoded_left, shape=(-1, 32, 64))
        # encoded_right = tf.reshape(encoded_right, shape=(-1, 32, 64))
        
        estm_acc_left = self.do_estimate(encoded_left, "gru_1", "attention_acc")
        estm_acc_right = self.do_estimate(encoded_right, "gru_2", "attention_acc")
        
        estm_voc_left = self.do_estimate(encoded_left, "gru_1", "attention_voc")
        estm_voc_right = self.do_estimate(encoded_right, "gru_2", "attention_voc")

        # tfmask
        tm_mask_voc_left = self.tf_mask(estm_voc_left, estm_acc_left) * self.input_left
        tm_mask_acc_left = self.tf_mask(estm_acc_left, estm_voc_left) * self.input_left

        tm_mask_voc_right = self.tf_mask(estm_voc_right, estm_acc_right) * self.input_right
        tm_mask_acc_right = self.tf_mask(estm_acc_right, estm_voc_right) * self.input_right

        result_acc_left = tf.reshape(tm_mask_acc_left, (-1, BATCH_SIZE, FRN_BIN))
        result_voc_left = tf.reshape(tm_mask_voc_left, (-1, BATCH_SIZE, FRN_BIN))

        result_acc_right = tf.reshape(tm_mask_acc_right, (-1, BATCH_SIZE, FRN_BIN))
        result_voc_right = tf.reshape(tm_mask_voc_right, (-1, BATCH_SIZE, FRN_BIN))

        self.acc_left = result_acc_left
        self.voc_left = result_voc_left
        self.acc_right = result_acc_right
        self.voc_right = result_voc_right

    def loss(self):
        # gamma = 0.2
        return tf.reduce_mean(\
              tf.square(self.acc_src_left - self.acc_left) + tf.square(self.voc_src_left - self.voc_left)\
              + tf.square(self.acc_src_right - self.acc_right) + tf.square(self.voc_src_right - self.voc_right)
            )
        # - gamma * tf.square(self.voc_src - self.acc) \
        # - gamma * tf.square(self.acc_src - self.voc), name="loss")

    def train(self, database, continue_index=0):
        self.build_model()
        loss_fn = self.loss()
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer().minimize(loss_fn, global_step=global_step)

        summary_op = self.summaries(loss_fn)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.load_state(sess, './model')

            writer = tf.summary.FileWriter('./graph', sess.graph)

            step = 0

            for index in range(50-continue_index):
                mixture, vocals, accompaniment = database.generate_batch_data(continue_index+index)
                mixture_left, mixture_right = mixture
                vocals_left, vocals_right = vocals
                accompaniment_left, accompaniment_right = accompaniment

                batch_len = 10

                times = math.ceil(len(mixture) / batch_len)

                for j in range(100):
                    for i in range(times):

                        step += 1

                        batch_mixture_left = mixture_left[i*batch_len : (i+1)*batch_len if (i+1)*batch_len < len(mixture_left) else len(mixture_left), :, :]
                        batch_mixture_right = mixture_right[i*batch_len : (i+1)*batch_len if (i+1)*batch_len < len(mixture_right) else len(mixture_right), :, :]
                        batch_voc_left = vocals_left[i*batch_len : (i+1)*batch_len if (i+1)*batch_len < len(vocals_left) else len(vocals_left), :, :]
                        batch_voc_right = vocals_right[i*batch_len : (i+1)*batch_len if (i+1)*batch_len < len(vocals_right) else len(vocals_right), :, :]
                        batch_acc_left = accompaniment_left[i*batch_len : (i+1)*batch_len if (i+1)*batch_len < len(accompaniment_left) else len(accompaniment_left), :, :]
                        batch_acc_right = accompaniment_right[i*batch_len : (i+1)*batch_len if (i+1)*batch_len < len(accompaniment_right) else len(accompaniment_right), :, :]

                        loss, _, summary = sess.run([loss_fn, optimizer, summary_op], feed_dict={
                                                                self.acc_src_left: batch_acc_left,
                                                                self.voc_src_left: batch_voc_left,
                                                                self.acc_src_right: batch_acc_right,
                                                                self.voc_src_right: batch_voc_right,
                                                                self.input_left: batch_mixture_left,
                                                                self.input_right: batch_mixture_right
                                                            })

                        print("Turn %s, Epoch %s, Batch %s : loss %s: " % (index, i, j, loss))

                        writer.add_summary(summary, global_step=step)

                del mixture, vocals, accompaniment
                database.release_batch_data()
                gc.collect()

                if index % 2 == 0:
                    self.save_state(sess, './model', continue_index+index)
                    
            writer.close()

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

                mask_acc = np.abs(acc) / (np.abs(acc) + np.abs(voc) + np.finfo(float).eps)
                mask_voc = 1. - mask_acc

                voc = batch_mixture * mask_acc
                acc = batch_mixture * mask_voc

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

    def summaries(self, loss):
        # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            # tf.summary.histogram(v.name, v)
            # tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('mixture_left', self.input_left)
        tf.summary.histogram('mixture_right', self.input_right)
        tf.summary.histogram('voc_left', self.voc_left)
        tf.summary.histogram('voc_right', self.voc_right)
        tf.summary.histogram('acc_left', self.acc_left)
        tf.summary.histogram('acc_right', self.acc_right)
        return tf.summary.merge_all()

nn = Network()
database = Database('train')
nn.train(database, 0)

# database2 = Database('test')
# nn.predict_with_model(database2, './model')
