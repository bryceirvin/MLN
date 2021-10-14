import librosa
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# use librosa to get intuition about what baseline features to explore, then code our own
# calling these without arguments will result in default settings

def get_spectrogram(y, n_fft=2048, hop_length=None, win_length=None,
                    window='hann', center=True, dtype=None, pad_mode='reflect'):
    spectrogram = np.abs(librosa.stft(y, n_fft, hop_length, win_length, window, center, dtype, pad_mode)).T
    return spectrogram


def get_mel_spectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None,
                        window='hann', center=True, pad_mode='reflect', power=2.0):
    mel_spectrogram = librosa.feature.melspectrogram(y, sr, S, n_fft, hop_length,
                                                     win_length, window, center, pad_mode, power).T
    return mel_spectrogram


def get_mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0,):
    mfcc = librosa.feature.mfcc(y, sr, S, n_mfcc, dct_type, norm, lifter).T
    return mfcc


def get_spectral_flatness(y=None, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann',
                          center=True, pad_mode='reflect', amin=1e-10, power=2.0):
    spectral_flatness = librosa.feature.spectral_flatness(y, S, n_fft, hop_length, win_length, window,
                                                          center, pad_mode=, amin, power)
    return spectral_flatness


def get_mean_spectrogram(y, n_fft=2048, hop_length=None, win_length=None,
                    window='hann', center=True, dtype=None, pad_mode='reflect'):
    spectrogram = np.abs(librosa.stft(y, n_fft, hop_length, win_length, window, center, dtype, pad_mode)).T
    return np.mean(spectrogram, axis=0)


def get_mean_mel_spectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None,
                        window='hann', center=True, pad_mode='reflect', power=2.0):
    mel_spectrogram = librosa.feature.melspectrogram(y, sr, S, n_fft, hop_length,
                                                     win_length, window, center, pad_mode, power).T
    return np.mean(mel_spectrogram, axis=0)


def get_mean_mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0,):
    mfcc = librosa.feature.mfcc(y, sr, S, n_mfcc, dct_type, norm, lifter).T
    return np.mean(mfcc, axis=0)


def get_mean_spectral_flatness(y=None, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann',
                          center=True, pad_mode='reflect', amin=1e-10, power=2.0):
    spectral_flatness = librosa.feature.spectral_flatness(y, S, n_fft, hop_length, win_length, window,
                                                          center, pad_mode=, amin, power)
    return np.mean(spectral_flatness, axis=0)

def get_z_score_scaler(features):
    scaler = StandardScaler()
    scaler.fit(features)

    return scaler

def get_min_max_scaler(features):
    scaler = MinMaxScaler()
    scaler.fit(features)

    return scaler

def normalize_features(features, scaler):
    return scaler.transform(features)





