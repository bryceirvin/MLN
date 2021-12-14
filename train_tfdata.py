import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
import tensorflow_io as tfio
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from sklearn.metrics import f1_score
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from utils.utils import *
from models.models import MelonQualityClassifier

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python train_tfdata.py <config_path> <data_csv_path> <results_dir>")
        sys.exit()

    config_path = sys.argv[1]
    data_csv_path = sys.argv[2]
    results_dir = sys.argv[3]

    with open(config_path, "rt") as f:
        config = yaml.load(f, Loader)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    melon_df = pd.read_csv(data_csv_path)

    # copy files into good or bad melon
    melon_df = pd.read_csv("Melon_Info.csv")
    for file_path in list(melon_df["path"]):
        ind = melon_df.index[melon_df['path'] == file_path].tolist()[0]
        sweetness = melon_df["sweetness"][ind]
        if sweetness == "not sweet" or sweetness == "somewhat sweet":
            dst = "melon_audio_categorical/bad/"
        elif sweetness == "sweet" or sweetness == "very sweet":
            dst = "melon_audio_categorical/good/"

        shutil.copy(file_path, os.path.join(dst, os.path.basename(file_path)))

    bad_files = os.listdir("melon_audio_categorical/bad")
    good_files = os.listdir("melon_audio_categorical/good")

    for i, path in enumerate(bad_files):
        bad_files[i] = os.path.join("melon_audio_categorical/bad", path)

    for i, path in enumerate(good_files):
        good_files[i] = os.path.join("melon_audio_categorical/good", path)

    if not config["data"]["use_knocks"]:
        melon_df = melon_df[melon_df["impulse"] == "Slap"]
        for file_path in list(slap_df["path"]):
            ind = slap_df.index[slap_df['path'] == file_path].tolist()[0]
            sweetness = slap_df["sweetness"][ind]
            if sweetness == "not sweet" or sweetness == "somewhat sweet":
                dst = "melon_audio_categorical_slaps/bad/"
            elif sweetness == "sweet" or sweetness == "very sweet":
                dst = "melon_audio_categorical_slaps/good/"

            shutil.copy(file_path, os.path.join(dst, os.path.basename(file_path)))

        bad_files_slaps = os.listdir("melon_audio_categorical_slaps/bad")
        good_files_slaps = os.listdir("melon_audio_categorical_slaps/good")
        for i, path in enumerate(bad_files_slaps):
            bad_files[i] = os.path.join("melon_audio_categorical_slaps/bad", path)

        for i, path in enumerate(good_files_slaps):
            good_files[i] = os.path.join("melon_audio_categorical_slaps/good", path)

    if not config["data"]["use_slaps"]:
        melon_df = melon_df[melon_df["impulse"] == "Knock"]
        for file_path in list(knock_df["path"]):
            ind = knock_df.index[knock_df['path'] == file_path].tolist()[0]
            sweetness = knock_df["sweetness"][ind]
            if sweetness == "not sweet" or sweetness == "somewhat sweet":
                dst = "melon_audio_categorical_knocks/bad/"
            elif sweetness == "sweet" or sweetness == "very sweet":
                dst = "melon_audio_categorical_knocks/good/"

            shutil.copy(file_path, os.path.join(dst, os.path.basename(file_path)))

        bad_files_knocks = os.listdir("melon_audio_categorical_knocks/bad")
        good_files_knocks = os.listdir("melon_audio_categorical_knocks/good")

        for i, path in enumerate(bad_files_knocks):
            bad_files[i] = os.path.join("melon_audio_categorical_knocks/bad", path)

        for i, path in enumerate(good_files_knocks):
            good_files[i] = os.path.join("melon_audio_categorical_knocks/good", path)

    melon_categories = ["bad", "good"]

    all_files = []
    all_files.extend(bad_files)
    all_files.extend(good_files)

    max_length = config["data"]["max_length"]

    train_files, test_files = train_test_split(all_files,
                                               test_size=1-config["data"]["train_ratio"],
                                               random_state=412,
                                               shuffle=True)
    val_files, test_files = train_test_split(test_files,
                                             test_size=config["data"]["val2test_ratio"],
                                             random_state=412,
                                             shuffle=True)

    print(f"{len(train_files)} train files")
    print(f"{len(val_files)} validation files")
    print(f"{len(test_files)} test files")

    AUTOTUNE = tf.data.AUTOTUNE

    files_ds = tf.data.Dataset.from_tensor_slices(train_files)

    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)

    train_ds = preprocess_dataset_melspectrogram(train_files, AUTOTUNE)
    val_ds = preprocess_dataset_melspectrogram(val_files, AUTOTUNE)
    test_ds = preprocess_dataset_melspectrogram(test_files, AUTOTUNE)

    batch_size = config["training"]["batch_size"]

    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape
        print(input_shape)

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)


    model_config = config["model"]
    MQC = MelonQualityClassifier(melon_categories, model_config)
    MQC.adapt_normalization_layer(train_ds)
    model = MQC.get_model(input_shape)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["num_epochs"],
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=config["training"]["early_stopping_patience"]),
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

    model.save(os.path.join(results_dir, "saved_model"))

    # evaluation
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels
    test_acc = np.sum(y_pred == y_true) / len(y_true)
    print(f"Accuracy on test data: {test_acc}")
    np.save(os.path.join(results_dir, "accuracy.npy"), test_acc)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Macro F1: {macro_f1}")
    print(f"Micro F1: {micro_f1}")
    print(f"Weighted F1: {weighted_f1}")

    np.save(os.path.join(results_dir, "macro_f1.npy"), macro_f1)
    np.save(os.path.join(results_dir, "micro_f1.npy"), micro_f1)
    np.save(os.path.join(results_dir, "weighted_f1.npy"), weighted_f1)



