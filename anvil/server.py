import sys

import anvil.server
from io import BytesIO
import requests
import tensorflow as tf
import urllib.request
import numpy as np
from miniaudio import SampleFormat, decode

sys.path.append("../")
from utils.utils import *

anvil.server.connect("ZUP7QJFF5O7Y4XGDC44HZG73-2TMXGJOGAYPKJJ2N")

@anvil.server.callable
def predict(url):
    
    audio_bytes = url.get_bytes()
    decoded_audio = decode(audio_bytes, nchannels=1, sample_rate=44100, output_format=SampleFormat.SIGNED32)
    
    mel_spectrogram = tf.expand_dims(get_melspectrogram(decoded_audio.samples), axis=0)
    
    model = tf.keras.models.load_model("../results/tfdata/saved_model")
    
    result = model.predict(mel_spectrogram)
    
    if np.argmax(result) == 0:
        message = "Sweet! What a Melon!"
    else:
        message = "Not sweet... Try another melon"
        
    return message

anvil.server.wait_forever()
    
