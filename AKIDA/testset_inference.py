from pathlib import Path
from zipfile import ZipFile

import tensorflow as tf
import numpy as np

import akida
import cnn2snn

from huggingface_hub import hf_hub_download

# ============================================================
# PATH
# ============================================================

AKIDA_DIR = Path(__file__).resolve().parent
ROOT_DIR = AKIDA_DIR.parent

DATASET_DIR = ROOT_DIR / "dataset" / "data"
CKPT_DIR = ROOT_DIR / "ckpt"
IMGS_DIR = ROOT_DIR / "imgs"
MODELS_DIR = ROOT_DIR / "models"

TEST_ZIP_PATH = DATASET_DIR / "test_frames.zip"

TEST_FRAMES_DIR = DATASET_DIR / "test_frames"

AKIDA_MODEL_PATH = CKPT_DIR / "FINAL_AKIDA_94.fbz"

HUGGINGFACE_DATASET = "neuromorphic-polito/Roshambo"


# ============================================================
# DEVICE CHECK
# ============================================================

print("##### DEVICE CHECK #####")

device = akida.devices()[0]
device.soc.power_measurement_enabled = True

print("SDK version:", akida.__version__)
print("Firmware version:", device.version)
print("Akida version:", cnn2snn.get_akida_version())
print("##### ###### #####")


# ============================================================
# DATASET
# ============================================================

def evaluate_dataset_akida(model_akida, test_ds, num_classes):
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in test_ds:
        x_batch = x_batch.numpy()
        y_batch = y_batch.numpy()

        acc = model_akida.evaluate(
            x_batch,
            y_batch,
            num_classes=num_classes
        )

        batch_size = len(y_batch)

        total_correct += acc * batch_size
        total_samples += batch_size

    final_acc = total_correct / total_samples

    print(
        "total_correct, total_samples:",
        total_correct,
        total_samples
    )

    return final_acc


def norm(img, lab):
    return (
        tf.cast(img, tf.uint8),
        tf.cast(lab, tf.uint8)
    )

# ============================================================
# DATASET PREPARATION
# ============================================================

if not TEST_ZIP_PATH.exists():
    print("Downloading Dataset", HUGGINGFACE_DATASET)

    zip_path = hf_hub_download(
        repo_id=HUGGINGFACE_DATASET,
        filename="test_frames.zip",
        repo_type="dataset",
        local_dir=DATASET_DIR,
    )
else:
    print("Dataset already available")

if not TEST_FRAMES_DIR.exists():
    print(f"Extracting dataset from: {TEST_ZIP_PATH}")

    with ZipFile(TEST_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(TEST_FRAMES_DIR.parent)

    print(f"Dataset extracted in: {TEST_FRAMES_DIR}")


test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_FRAMES_DIR,
    # mapping
    # paper      -> 0
    # scissors   -> 1
    # rock       -> 2
    # background -> 3
    class_names=[
        "0",
        "1",
        "2",
        "3",
    ],

    image_size=(64, 64),
    color_mode="grayscale",
    batch_size=512,
)

test_ds = test_ds.map(norm)


# ============================================================
# AKIDA INFERENCE
# ============================================================

device_akida = akida.devices()[0]
device_akida.soc.power_measurement_enabled = True


if not AKIDA_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modello Akida non trovato: {AKIDA_MODEL_PATH}"
    )


akida_model = akida.Model(str(AKIDA_MODEL_PATH))

akida_model.map(device_akida)
akida_model.summary()

print("Model version:", akida_model.ip_version, "\n")

myBackend = akida_model.sequences[0].backend
print(f"Model backend: {myBackend}")


# ============================================================
# WARM-UP / POWER
# ============================================================

single_batch = next(iter(test_ds))

img_batch, lab_batch = single_batch

_ = akida_model.forward(img_batch.numpy())

floor_power = device_akida.soc.power_meter.floor

print(f"Floor power: {floor_power:.2f} mW")
print("First batch:", akida_model.statistics)


# ============================================================
# EVALUATION
# ============================================================
acc_batched = evaluate_dataset_akida(
    akida_model,
    test_ds,
    num_classes=4
)

print(f"Akida accuracy: {acc_batched:.4f}")