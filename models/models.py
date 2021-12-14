import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.io.wavfile import read
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence, to_categorical


class MelonDataset(Sequence):

    def __init__(self, data_df, batch_size, max_length, binarize, shuffle=True):
        self.melon_df = data_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.binarize = binarize
        self.len = self.__len__()

        if self.binarize:
            self.num_classes = 2
        else:
            self.num_classes = 4

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.melon_df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.melon_df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def getitem(self, index):
        return self.__getitem__(index)

    def __getitem__(self, idx):

        indexes = self.indexes[
                  idx * self.batch_size:(idx + 1) * self.batch_size]

        inputs = []
        labels = []
        for i in indexes:
            path, melon, mic_point, impulse, impulse_point, sweetness = self.melon_df.iloc[i]
            #audio_file = tf.io.read_file(path)
            #audio, _ = tf.audio.decode_wav(contents=audio_file)
            _, int_audio = read(path)
            audio = np.array([np.float32((sample >> 2) / 32768.0) for sample in int_audio])
            if len(audio) < self.max_length:
                paddings = [[0, self.max_length-len(audio)]]
                audio = tf.pad(audio, paddings)

            inputs.append(audio)

            if sweetness == "not sweet":
                if self.binarize:
                    label = 0
                else:
                    label = 0
            elif sweetness == "somewhat sweet":
                if self.binarize:
                    label = 0
                else:
                    label = 1
            elif sweetness == "sweet":
                if self.binarize:
                    label = 1
                else:
                    label = 2
            elif sweetness == "very sweet":
                if self.binarize:
                    label = 1
                else:
                    label = 3

            labels.append(to_categorical(label, num_classes=self.num_classes))

        inputs = np.stack(inputs)
        # print("Inputs shape:", inputs.shape)

        labels = np.stack(labels)
        # print("Labels shape:", labels.shape)

        return inputs, labels


class MelonQualityClassifier:
    def __init__(self, melon_categories, model_config):
        self.num_classes = len(melon_categories)
        self.normalization_layer = Normalization()
        self.filters = model_config["filters"]
        self.kernel_size = model_config["kernel"]
        self.activation = model_config["activation"]
        self.dense_dim = model_config["dense_dim"]

    def adapt_normalization_layer(self, ds):
        self.normalization_layer.adapt(data=ds.map(map_func=lambda spec, label: spec))

    def get_model(self, input_shape):

        inputs = Input(input_shape)
        x = Resizing(32, 32)(inputs)
        x = self.normalization_layer(x)
        for filter_size in self.filters:
            x = Conv2D(filter_size, self.kernel_size, activation=self.activation)(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(self.dense_dim, activation=self.activation)(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes)(x)

        model = Model(inputs=inputs, outputs=outputs, name="MelonQualityClassifer")

        return model

class ExtractFeatures(Layer):
    def __init__(self, sampling_rate, frame_length, frame_step, fft_length, lower_edge_hz,
                 upper_edge_hz, eps, num_mel_bins, use_mfccs, num_mfccs):
        super(ExtractFeatures, self).__init__()
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.lower_edge_hz = lower_edge_hz
        self.upper_edge_hz = upper_edge_hz
        self.num_mel_bins = num_mel_bins
        self.eps = eps
        self.use_mfccs = use_mfccs
        self.num_mfccs = num_mfccs

    def get_config(self):
        config = super().get_config()
        config.update({
            "sampling_rate": self.sampling_rate,
            "frame_length": self.frame_length,
            "frame_step": self.frame_step,
            "fft_length": self.fft_length,
            "lower_edge_hz": self.lower_edge_hz,
            "upper_edge_hz": self.upper_edge_hz,
            "num_mel_bins": self.num_mel_bins,
            "eps": self.eps,
            "use_mfccs": self.use_mfccs,
            "num_mfccss": self.num_mfccs,
        })

        return config

    def call(self, inputs):
        stfts = tf.signal.stft(inputs,
                               frame_length=self.frame_length,
                               frame_step=self.frame_step,
                               fft_length=self.fft_length,
                               )
        mag_stfts = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, self.sampling_rate, self.lower_edge_hz,
            self.upper_edge_hz)
        mel_spectrograms = tf.tensordot(
            mag_stfts, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(mag_stfts.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # log mel spectrogram
        features = tf.math.log(mel_spectrograms + self.eps)

        if self.use_mfccs:
            features = tf.signal.mfccs_from_log_mel_spectrograms(features)[..., :self.num_mfccs]

        return features[..., tf.newaxis]




def get_cnn(input_shape, filters, kernel, padding, activation, num_conv_blocks, binarize,
            sampling_rate, frame_length, frame_step, fft_length, lower_edge_hz, upper_edge_hz,
            eps, num_mel_bins, use_mfccs, num_mfccs):

    inputs = Input(input_shape)
    x = ExtractFeatures(sampling_rate, frame_length, frame_step, fft_length, lower_edge_hz,
                        upper_edge_hz, eps, num_mel_bins, use_mfccs, num_mfccs)(inputs)
    x = BatchNormalization()(x)

    for i in range(num_conv_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel, padding=padding, activation=activation)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(64, activation=activation)(x)

    if binarize:
        n_outputs = 2
    else:
        n_outputs = 4

    outputs = Dense(n_outputs, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_cnn_v2(input_shape, filters, kernel, padding, activation, num_conv_blocks, binarize,
            sampling_rate, frame_length, frame_step, fft_length, lower_edge_hz, upper_edge_hz,
            eps, num_mel_bins, use_mfccs, num_mfccs):

    inputs = Input(input_shape)
    x = ExtractFeatures(sampling_rate, frame_length, frame_step, fft_length, lower_edge_hz,
                        upper_edge_hz, eps, num_mel_bins, use_mfccs, num_mfccs)(inputs)
    x = BatchNormalization()(x)

    for i in range(num_conv_blocks):
        x = Conv2D(filters=filters, kernel_size=kernel, padding=padding, activation=activation)(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(128, activation=activation)(x)
    x = Dropout(0.5)(x)

    if binarize:
        n_outputs = 2
    else:
        n_outputs = 4

    outputs = Dense(n_outputs, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model







