import akida
import quantizeml
from quantizeml.models import quantize, QuantizationParams
import cnn2snn
from cnn2snn import convert, set_akida_version, AkidaVersion
from akida_models.sparsity import compute_sparsity

from CPU.utils.preprocessing import norm
import os
import tensorflow as tf
import time

print("##### DEVICE CHECK #####")
os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = "v1"


device = akida.devices()[0]
device.soc.power_measurement_enabled = True

#Check SDK (software) version
print("SDK version:", akida.__version__)
print("Firmware version:", device.version)
print("Akida version", cnn2snn.get_akida_version())
print("##### ###### #####")




test_ds = tf.keras.utils.image_dataset_from_directory(
    "../dataset/data/test_frames",
    image_size=(64,64),
    color_mode="grayscale",
    batch_size=1024
)

test_ds = test_ds.map(norm)

single_batch = next(iter(test_ds))
img_batch, lab_batch = single_batch






#AKIDA INFERENCE AND POWER MEASUREMENT
device_akida = akida.devices()[0]
device_akida.soc.power_measurement_enabled = True


akida_model = akida.Model("../ckpt/FINAL_AKIDA_94.fbz")
akida_model.map(device_akida, hw_only=True, mode=akida.MapMode.AllNps)

akida_model.summary()
print('Model version: ', akida_model.ip_version, '\n')

myBackend = akida_model.sequences[0].backend
print(f"Model backend: {myBackend}")


img_batch_np = img_batch.numpy()
lab_batch_np = lab_batch.numpy()
print("img_batch shape:", img_batch_np.shape, "lab_batch shape:", lab_batch_np.shape)


t0 = time.perf_counter()
predictions = akida_model.forward(img_batch_np)
t1 = time.perf_counter()

akida_duration_s = t1 - t0



dict_sparsity = compute_sparsity(
    model = akida_model,
    samples = img_batch_np,
    batch_size = 1024)

print("Sparsity dict full: ", dict_sparsity)

