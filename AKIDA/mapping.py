import tensorflow as tf
import numpy as np
from dataset import caffe_pb2, dataset
from models import *
from CPU.utils.accumulate_to_image import *
import cv2 as cv
import os

import akida
import quantizeml
from quantizeml.models import quantize, QuantizationParams
import cnn2snn
from cnn2snn import convert, set_akida_version, AkidaVersion
from akida_models.sparsity import compute_sparsity


#Uses categories {paper: 0, rock: 2, scissors: 1, background: 3} for 

print("##### DEVICE CHECK #####")
os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = "v1"

device = akida.devices()[0]
device.soc.power_measurement_enabled = True

#Check SDK (software) version
print("SDK version:", akida.__version__)
print("Firmware version:", device.version)
print("Akida version", cnn2snn.get_akida_version())
print("##### ###### #####")


def norm(img,lab):
    return tf.cast(img, tf.uint8), tf.cast(lab, tf.uint8),


def evaluate_dataset_akida(model_akida, test_ds, num_classes):
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in test_ds: #the batch size is defined when constructin this set
        # Convert to numpy with correct dtypes

        # Evaluate this batch (Akida returns accuracy for this chunk)
        acc = model_akida.evaluate(x_batch, y_batch, num_classes=num_classes)

        # Accumulate weighted by batch size
        batch_size = len(y_batch)
        total_correct += acc * batch_size
        total_samples += batch_size

    final_acc = total_correct / total_samples
    print("total_correct, total_samples:", total_correct, total_samples)
    return final_acc



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



#LOAD MODEL KERAS
model = tf.keras.models.load_model("../ckpt/same_epoch_40.keras")
model.summary()

#_, acc = model.evaluate(test_ds, verbose=2)
#print("Accuracy imported:", acc) 96





#QUANTIZATION FOR AKIDA

tf.config.run_functions_eagerly(False)

qparams = QuantizationParams(input_weight_bits=8, 
                            weight_bits=4, 
                            activation_bits=4,
                            per_tensor_activations=True) #no need calibration on data since we do QTA

model_quantized = quantize(model, qparams=qparams)

model_quantized.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy'])


hist_qt = model_quantized.fit(train_ds, 
                validation_data=val_ds, 
                epochs=4, 
                batch_size=128, 
                verbose=2)


_,acc = model_quantized.evaluate(test_ds, verbose=2)
print("After QTA: ", acc)

#model_quantized.save(root+"epoch_30_QTA.keras")






# AKIDA CONVERSION AND MAPPING

model = tf.keras.models.load_model("../ckpt/same_QTA_def.keras")
model.compile(metrics=['accuracy'])

#_, acc = model.evaluate(test_ds, verbose=2) #96.2 
#print("Accuracy QTA imported keras:", acc) 



model_akida = convert(model)
model_akida.summary()
print('Model version: ', model_akida.ip_version, '\n')


device_akida = akida.devices()[0]
model_akida.map(device_akida, hw_only=True, mode=akida.MapMode.Minimal)
model_akida.summary()


acc_akd = evaluate_dataset_akida(model_akida, test_ds, 4)
print("Accuracy on Akida:", acc_akd)

#model_akida.save("FINAL_AKIDA_94.fbz")
