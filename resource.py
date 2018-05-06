'''
File: \resource.py
Project: AudioSourceSeparation
Created Date: Tuesday April 24th 2018
Author: Huisama
-----
Last Modified: Monday May 7th 2018 12:26:02 am
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import numpy as np
import musdb
import wave
import math
from scipy.io.wavfile import write

mus = musdb.DB(root_dir="./musdb18")

SOURCE_SEG_LEN = 4410
SFFT_LEN = 441
SAMPLE_LEN = SOURCE_SEG_LEN // SFFT_LEN
SFFT_BIN = 1024
BATCH_SIZE = 4

class Database(object):
    '''
        Database class to provide formatted audio data for training
    '''
    def __init__(self, subset, segment_length=SOURCE_SEG_LEN):
        self.subset = subset
        self.segment_length = segment_length
        self.raw_tracks = mus.load_mus_tracks(subsets=self.subset)

        self.batch_index = 0
        self.batch_size = 2

    def _split(self, data):
        list = []
        for t in data:
            result = np.array_split(t, t.shape[0] / self.segment_length)
            list.append(result)
        return list

    def _generate_batch_data(self, data, flatten=False):
        if flatten:
            result = []
            for song in data:
                for seg in song:
                    sfft_left, sfft_right = DataOperation.stft(seg)
                    # sfft = np.stack((sfft_left.real, sfft_left.imag, sfft_right.real, sfft_right.imag))
                    sfft = np.stack((sfft_left.real, sfft_right.real))
                    sfft = np.transpose(sfft, (1, 0, 2))
                    sfft = np.reshape(sfft, (SAMPLE_LEN, -1))

                    result.append(sfft)
            combined_arr = []
            for i in range(len(result) // BATCH_SIZE):
                comb = np.stack(result[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                comb = np.reshape(comb, (-1, 2048))
                combined_arr.append(comb)
            return combined_arr

        else:
            result = []
            for song in data:
                for seg in song:
                    sfft_left, sfft_right = DataOperation.stft(seg)
                    # sfft = np.stack((sfft_left.real, sfft_left.imag, sfft_right.real, sfft_right.imag))
                    sfft = np.stack((sfft_left.real, sfft_right.real))
                    sfft = np.transpose(sfft, (1, 0, 2))
                    
                    # for sfft_seg in sfft:
                    #     sfft_seg = sfft_seg[:, :, np.newaxis]
                    result.append(sfft)
            combined_arr = []
            for i in range(len(result) // BATCH_SIZE):
                comb = np.stack(result[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                comb = np.reshape(comb, (-1, 64, 32))
                combined_arr.append(comb)
            return combined_arr

    def _pad_array(self, array):
        zero_count = self.segment_length * (math.ceil(array.shape[0] / self.segment_length)) - array.shape[0]
        result = np.pad(array, ((0, zero_count), (0, 0)), 'constant')
        # print(result.shape)
        return result

    def _get_batch_data(self):
        batch_tracks = self.raw_tracks[self.batch_index: self.batch_index+self.batch_size]
        print("\nBatch index from %s to %s\n" % (self.batch_index, self.batch_index+2))
                
        self.batch_index += self.batch_size

        self.batch_index %= len(self.raw_tracks)

        mixture = self._split([self._pad_array(track.audio) for track in batch_tracks])
        vocals = self._split([self._pad_array(track.targets['vocals'].audio) for track in batch_tracks])
        accompaniment = self._split([self._pad_array(track.targets['accompaniment'].audio) for track in batch_tracks])

        return mixture, vocals, accompaniment

    def generate_batch_data(self):
        mixture, vocals, accompaniment = self._get_batch_data()

        new_mixture = self._generate_batch_data(mixture)
        new_vocals = self._generate_batch_data(vocals, True)
        new_accompaniment = self._generate_batch_data(accompaniment, True)

        return new_mixture, new_vocals, new_accompaniment

class DataOperation:
    @classmethod
    def _split_data(cls, data, duration):
        result = np.array_split(data, data.shape[0] / duration)
        return result

    @classmethod
    def stft(cls, data, duration=SFFT_LEN):
        left = data[:, 0]
        right = data[:, 1]

        sfft_left = [np.fft.fft(seg, SFFT_BIN).real for seg in cls._split_data(left, duration)]
        sfft_right = [np.fft.fft(seg, SFFT_BIN).real for seg in cls._split_data(right, duration)]

        arrl = np.array(sfft_left)
        arrr = np.array(sfft_right)

        return arrl, arrr

    @classmethod
    def istft(cls, sfft_left, sfft_right):
        pass

db = Database("train")
db.generate_batch_data()
