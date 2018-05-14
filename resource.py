'''
File: \resource.py
Project: AudioSourceSeparation
Created Date: Tuesday April 24th 2018
Author: Huisama
-----
Last Modified: Monday May 14th 2018 9:07:56 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import numpy as np
import musdb
import wave
import math
from scipy.io.wavfile import write
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

mus = musdb.DB(root_dir="./musdb18")

BATCH_SIZE = 32
BATCH_SONG_SIZE = 1
FRN_SAMPLE = 512
FRN_BIN = FRN_SAMPLE // 2 + 1

class Database(object):
    '''
        Database class to provide formatted audio data for training
    '''
    def __init__(self, subset):
        self.subset = subset
        self.raw_tracks = mus.load_mus_tracks(subsets=self.subset)

        self.batch_index = 0
        self.batch_size = BATCH_SONG_SIZE

    def _generate_padding_data(self, data):
        # pad zero array
        data = np.transpose(data, (1, 0))

        points = data.shape[0]
        remain = BATCH_SIZE * math.ceil((points / BATCH_SIZE)) - points

        if remain == 0:
            return data
        remain_arr = np.zeros((remain, data.shape[1]))
        data = np.concatenate((data, remain_arr))
        return data

    def _generate_data_set_for_nn(self, data):
        result = []
        for raw in data:
            #Do sfft
            left = raw[:, 0]
            right = raw[:, 1]

            _, _, Zxxl = stft(x=left, fs=0.1, nperseg=FRN_SAMPLE)
            _, _, Zxxr = stft(x=right, fs=0.1, nperseg=FRN_SAMPLE)

            # Padding
            Zxxl_seq = self._generate_padding_data(Zxxl.real)
            Zxxr_seq = self._generate_padding_data(Zxxr.real)

            # Stack
            stacked = np.stack((Zxxl_seq, Zxxr_seq))
            stacked = np.transpose(stacked, (1, 0, 2))

            # reshape
            reshaped = np.reshape(stacked, (-1, BATCH_SIZE, 2, stacked.shape[2]))

            result.append(reshaped)

        return np.concatenate(result)        

    def generate_batch_data(self, continue_index=0):
        self.batch_index = continue_index

        tracks = self.raw_tracks[self.batch_index:self.batch_index+self.batch_size]

        print('From %s to %s' % (self.batch_index, self.batch_index + BATCH_SONG_SIZE - 1))

        self.batch_index += BATCH_SONG_SIZE
        self.batch_index %= 50

        mixture = [track.audio for track in tracks]
        vocals = [track.targets['vocals'].audio for track in tracks]
        accompaniments = [track.targets['accompaniment'].audio for track in tracks]

        mixture_dataset = self._generate_data_set_for_nn(mixture)
        vocals_dataset = self._generate_data_set_for_nn(vocals)
        accompaniments_dataset = self._generate_data_set_for_nn(accompaniments)

        return mixture_dataset, vocals_dataset, accompaniments_dataset

    def generate_one_batch_data_for_test(self, index, flatten = False):
        mixture = [self.raw_tracks[index].audio]
        mixture_dataset = self._generate_data_set_for_nn(mixture)
        # vocals_dataset = self._generate_data_set_for_nn(mixture)
        # accompaniments_dataset = self._generate_data_set_for_nn(mixture)

        return mixture_dataset #, vocals_dataset, accompaniments_dataset

class DataOperation:
    @classmethod
    def data_to_wav(cls, data, filepath):
        write(filepath, 44100, data)

    @classmethod
    def reformat_data(cls, data):
        data = np.transpose(data, (2, 3, 0, 1))
        data = np.reshape(data, (2, FRN_BIN, -1))
        return data

    @classmethod
    def nn_output_to_wav(cls, data, filepath):
        result = cls.reformat_data(data)

        left = result[0, :, :]
        right = result[1, :, :]

        isftslr = istft(Zxx=left, fs=0.1, nperseg=FRN_SAMPLE)
        isftsrr = istft(Zxx=right, fs=0.1, nperseg=FRN_SAMPLE)

        stacked = np.stack((isftslr[1].real, isftsrr[1].real))
        stacked = np.transpose(stacked, (1, 0))

        DataOperation.data_to_wav(stacked, filepath)

# db = Database('sample')
# db.generate_batch_data()