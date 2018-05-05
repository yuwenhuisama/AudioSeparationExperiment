'''
File: \resource.py
Project: AudioSourceSeparation
Created Date: Tuesday April 24th 2018
Author: Huisama
-----
Last Modified: Sat May 05 2018
Modified By: huisama
-----
Copyright (c) 2018 Hui
'''

import numpy as np
import musdb
import wave
from scipy.io.wavfile import write

mus = musdb.DB(root_dir="./musdb18")

SOURCE_SEG_LEN = 4410
SFFT_LEN = 441
SAMPLE_LEN = SOURCE_SEG_LEN // SFFT_LEN
SFFT_BIN = 1024

class Database(object):
    '''
        Database class to provide formatted audio data for training
    '''
    def __init__(self, subset, segment_length=SOURCE_SEG_LEN):
        self.subset = subset
        self.segment_length = segment_length
        self.raw_tracks = mus.load_mus_tracks(subsets=self.subset)
        self.mixture = self.get_mixture_data()
        self.vocals = self.get_vocal_data()
        self.accompaniment = self.get_accompaniment_data()

    def get_mixture_data(self):
        result = self._split([track.audio for track in self.raw_tracks])
        return result

    def get_vocal_data(self):
        result = self._split([track.targets['vocals'].audio for track in self.raw_tracks])
        return result        

    def get_accompaniment_data(self):
        result = self._split([track.targets['accompaniment'].audio for track in self.raw_tracks])
        return result

    def _split(self, data):
        list = []
        for t in data:
            result = np.array_split(t, t.shape[0] / self.segment_length)
            list.append(result)
        return list

    def _generate_batch_data(self, data, flatten=False):
        result = []


        if flatten:
            # if flatten:
            # sfft_seg = np.reshape(sfft_seg, (-1,))
            for song in data:
                song_result = [[] for _ in range(SAMPLE_LEN)]
                for seg in song:
                    sfft_left, sfft_right = DataOperation.stft(seg)
                    sfft = np.stack((sfft_left.real, sfft_left.imag, sfft_right.real, sfft_right.imag))
                    sfft = np.transpose(sfft, (1, 0, 2))
                    sfft = np.reshape(sfft, (SAMPLE_LEN, -1))

                    index = 0
                    for sfft_seg in sfft:
                        song_result[index].append(sfft_seg)
                        index += 1
                # Todo: change to all songs
                result = song_result

        else:
            for song in data:
                song_result = [[] for _ in range(SAMPLE_LEN)]
                for seg in song:
                    sfft_left, sfft_right = DataOperation.stft(seg)
                    sfft = np.stack((sfft_left.real, sfft_left.imag, sfft_right.real, sfft_right.imag))
                    sfft = np.transpose(sfft, (1, 0, 2))

                    index = 0
                    for sfft_seg in sfft:
                        sfft_seg = sfft_seg[:, :, np.newaxis]
                        song_result[index].append(sfft_seg)
                        index += 1

                # Todo: change to all songs
                result = song_result
                
        for i in range(len(result)):
            result[i] = np.array(result[i])

        return result

    def generate_batch_data(self):
        mixture = self._generate_batch_data(self.mixture)
        vocal = self._generate_batch_data(self.vocals, True)
        accompaniment = self._generate_batch_data(self.accompaniment, True)

        return mixture, vocal, accompaniment
        

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

# db = Database("sample")
# db.generate_batch_data()
