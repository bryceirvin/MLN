import os

import tensorflow as tf
import tensorflow_io as tfio

from scipy.io.wavfile import read
from tqdm import tqdm

def get_max_length(list_audio_files):
    max_length = 0
    for audio_file in list_audio_files:
        sr, y = read(audio_file)
        if len(y) > max_length:
            max_length = len(y)

    return max_length


def decode_audio(audio_binary):
    audio = tfio.audio.decode_wav(audio_binary, dtype=tf.int32)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2]

def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform, max_length):
    input_len = max_length
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([max_length] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=2048, frame_step=512)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_melspectrogram(waveform, max_length=22050):
    sample_rate = 44100
    spectrogram = tf.squeeze(get_spectrogram(waveform, max_length), axis=-1)
    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 22000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms[..., tf.newaxis]

def get_spectrogram_and_label_id(audio, label, max_length=22050, melon_categories = ["bad", "good"]):
    spectrogram = get_spectrogram(audio, max_length)
    label_id = tf.argmax(label == melon_categories)
    return spectrogram, label_id

def get_melspectrogram_and_label_id(audio, label, max_length=22050, melon_categories = ["bad", "good"]):
    spectrogram = get_melspectrogram(audio)
    label_id = tf.argmax(label == melon_categories)
    return spectrogram, label_id

def preprocess_dataset(files, AUTOTUNE):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
    return output_ds

def preprocess_dataset_melspectrogram(files, AUTOTUNE):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
      map_func=get_melspectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
    return output_ds