import os
import sys

# Supress Tensorflow messages and unblock GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tqdm import tqdm

from models.models import get_cnn, get_cnn_v2, MelonDataset
from utils.utils import get_max_length


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python train.py <config_path> <data_csv_path> <results_dir>")
        sys.exit()

    config_path = sys.argv[1]
    data_csv_path = sys.argv[2]
    results_dir = sys.argv[3]

    with open(config_path, "rt") as f:
        config = yaml.load(f, Loader)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    melon_df = pd.read_csv(data_csv_path)

    if not config["data"]["use_knocks"]:
        melon_df = melon_df[melon_df["impulse"] == "Slap"]

    if not config["data"]["use_slaps"]:
        melon_df = melon_df[melon_df["impulse"] == "Knock"]

    # loop through files to get max length
    max_length = get_max_length(list(melon_df["path"]))

    train, test = train_test_split(melon_df, test_size=1-config["data"]["train_ratio"], random_state=412, shuffle=True)
    val, test = train_test_split(test, test_size=config["data"]["val2test_ratio"], random_state=412, shuffle=True)

    print(f"{len(train)} train files")
    print(f"{len(val)} validation files")
    print(f"{len(test)} test files")

    # save training and test files
    train.to_csv("data/train.csv")
    val.to_csv("data/val.csv")
    test.to_csv("data/test.csv")

    batch_size = config["training"]["batch_size"]
    binarize = config["data"]["binarize"]

    # create datasets
    train_ds = MelonDataset(train, batch_size, max_length, binarize)
    val_ds = MelonDataset(val, batch_size, max_length, binarize)
    test_ds = MelonDataset(test, batch_size, max_length, binarize)

    # get model info
    filters = config["model"]["filters"]
    kernel = config["model"]["kernel"]
    padding = config["model"]["padding"]
    activation = config["model"]["activation"]
    num_conv_blocks = config["model"]["num_conv_blocks"]

    feature = config["data"]["feature"]
    if feature == "mfcc":
        use_mfccs = True
    else:
        use_mfccs = False
    sampling_rate = config["features"]["sampling_rate"]
    frame_length = config["features"]["frame_length"]
    frame_step = config["features"]["frame_step"]
    fft_length = config["features"]["fft_length"]
    lower_edge_hz = config["features"]["lower_edge_hz"]
    upper_edge_hz = config["features"]["upper_edge_hz"]
    num_mel_bins = config["features"]["num_mel_bins"]
    eps = config["features"]["eps"]

    num_mfccs = config["features"]["num_mfccs"]

    input_shape = (max_length,)
    model = get_cnn_v2(input_shape, filters, kernel, padding, activation, num_conv_blocks, binarize,
            sampling_rate, frame_length, frame_step, fft_length, lower_edge_hz, upper_edge_hz,
            eps, num_mel_bins, use_mfccs, num_mfccs)

    model.summary()

    train_steps = train_ds.len
    validation_steps = val_ds.len

    # learning rate, epochs, early stopping
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["early_stopping_patience"]

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"],
                  )

    checkpointer = ModelCheckpoint(filepath=os.path.join(results_dir, 'saved_model'), verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    logger = TensorBoard(log_dir=os.path.join(results_dir, "logs"))

    workers = multiprocessing.cpu_count()

    print("Training!")
    history = model.fit(train_ds,
                        steps_per_epoch=train_steps,
                        workers=workers,
                        use_multiprocessing=False,
                        epochs=num_epochs,
                        verbose=1,
                        validation_data=val_ds,
                        validation_steps=validation_steps,
                        callbacks=[checkpointer,
                                   earlystopper,
                                   logger,
                                   ],
                        )

    plt.figure()
    plt.plot(history.history['loss'], label="train")
    plt.plot(history.history['val_loss'], label="val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss.png'))
    plt.close('all')

    plt.figure()
    plt.plot(np.log10(history.history['loss']), label="train")
    plt.plot(np.log10(history.history['val_loss']), label="val")
    plt.xlabel("epochs")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'log_loss.png'))
    plt.close('all')

    plt.figure()
    plt.plot(np.log10(history.history['accuracy']), label="train")
    plt.plot(np.log10(history.history['val_accuracy']), label="val")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'accuracy.png'))
    plt.close('all')

    model.save_weights(os.path.join(results_dir, "model.h5"))

    model.save(os.path.join(results_dir, "saved_model"))

    model.evaluate(test_ds)

    print("Done training")





