import os
import sys

import tensorflow as tf
import lmdb
from . import caffe_pb2
import numpy as np
from tqdm import tqdm
import cv2 as cv



def lmdb_to_arrays(lmdb_path, max_items=None):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        
        # Primo giro: conta le entry se max_items non è dato
        if max_items is None:
            max_items = sum(1 for _ in cursor)
            cursor.first()  # reset cursore

        # Avanzamento
        images = None
        labels = np.empty((max_items,), dtype=np.int32)

        for i, (key, val) in enumerate(cursor):
            if i >= max_items:
                break

            datum = caffe_pb2.Datum()
            datum.ParseFromString(val)
            flat = np.frombuffer(datum.data, dtype=np.uint8)
            img = flat.reshape(datum.channels, datum.height, datum.width)
            img = np.transpose(img, (1, 2, 0))  # H, W, C

            if images is None:
                # inizializzazione array finale
                H, W, C = img.shape
                images = np.empty((max_items, H, W, C), dtype=np.uint8)

            images[i] = img
            labels[i] = datum.label

    env.close()
    return images, labels



def make_dataset(images, labels, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_images_to_folder(images, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (img, label) in tqdm(enumerate(zip(images, labels)), total=len(images)):
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{i}.png")
        #Image.fromarray(img.squeeze(), mode="L").save(img_path)
        cv.imwrite(img_path, img.squeeze())



def make_datasets_from_disk(data_dir, batch_size=64, validation_split=0.2, seed=42, image_size=(64, 64)):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        color_mode="grayscale",
        batch_size=batch_size
    )

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
