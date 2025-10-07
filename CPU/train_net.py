import tensorflow as tf
import numpy as np
from dataset import caffe_pb2, dataset
from rockscissorpaper.models.models import *
from CPU.utils.accumulate_to_image import *
import cv2 as cv
from CPU.utils.preprocessing import norm




#dataset preparing

train_ds, val_ds = dataset.make_datasets_from_disk("../dataset/data/train_frames")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "../dataset/data/test_frames",
    image_size=(64,64),
    color_mode="grayscale",
    batch_size=128
)

train_ds = train_ds.map(norm)
val_ds = val_ds.map(norm)
test_ds = test_ds.map(norm)




#TRAIN MODEL FROM SCRATCH

model = roshambo_net()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy'])
model.summary()
print("model built")


#model training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = "../ckpt/epoch_{epoch:02d}.keras",
            save_freq = "epoch",
            save_weights_only = False
)


hist = model.fit(train_ds, 
                validation_data=val_ds, 
                epochs=30, 
                batch_size=128, 
                verbose=2,
                callbacks = checkpoint_callback)

print("success fin")
