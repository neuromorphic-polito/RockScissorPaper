from huggingface_hub import hf_hub_download
from pathlib import Path
from zipfile import ZipFile

import akida
import quantizeml
from quantizeml.models import quantize, QuantizationParams
import cnn2snn
from cnn2snn import convert, set_akida_version, AkidaVersion
from akida_models.sparsity import compute_sparsity

from dataset import caffe_pb2, dataset

from nets import *

CPU_DIR = Path(__file__).resolve().parent
ROOT_DIR = CPU_DIR.parent

DATASET_DIR = ROOT_DIR / "dataset" / "data"
CKPT_DIR = ROOT_DIR / "ckpt"
HUGGINGFACE_DATASET = "neuromorphic-polito/Roshambo"

TEST_ZIP_PATH = DATASET_DIR / "test_frames.zip"
TEST_FRAMES_DIR = DATASET_DIR / "test_frames"

TRAIN_ZIP_PATH = DATASET_DIR / "train_frames.zip"
TRAIN_FRAMES_DIR = DATASET_DIR / "train_frames"

"""
# ============================================================
# DATASET PREPARATION
# ============================================================

if not TEST_ZIP_PATH.exists():
    print("Downloading TEST Dataset", HUGGINGFACE_DATASET)

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



if not TRAIN_ZIP_PATH.exists():
    print("Downloading TRAIN Dataset", HUGGINGFACE_DATASET)

    zip_path = hf_hub_download(
        repo_id=HUGGINGFACE_DATASET,
        filename="train_frames.zip",
        repo_type="dataset",
        local_dir=DATASET_DIR,
    )
else:
    print("Dataset already available")


if not TRAIN_FRAMES_DIR.exists():
    print(f"Extracting dataset from: {TRAIN_ZIP_PATH}")

    with ZipFile(TRAIN_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(TRAIN_FRAMES_DIR.parent)

    print(f"Dataset extracted in: {TRAIN_FRAMES_DIR}")
"""




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--input_weight_bits", type=int, required=True)
    parser.add_argument("--weight_bits", type=int, required=True)
    parser.add_argument("--activation_bits", type=int, required=True)
    args = parser.parse_args()


    train_ds, val_ds = dataset.make_datasets_from_disk(TRAIN_FRAMES_DIR)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_FRAMES_DIR,
        image_size=(64,64),
        color_mode="grayscale",
        batch_size=128
    )


    #load model pretrained (check if correct ckpt)
    model = roshambo_net()
    model = tf.keras.models.load_model(CKPT_DIR/"same_epoch_40.keras")
    model.summary()

    #QUANTIZATION AND AKIDA MAPPING
    with set_akida_version(AkidaVersion.v1): #no need since already set at the beginning of the script

        tf.config.run_functions_eagerly(False)
        qparams = QuantizationParams(args.input_weight_bits, 
                                    args.weight_bits, 
                                    args.activation_bits,
                                    per_tensor_activations=True) #no need calibration on data since we do QTA


        def norm(img,lab):
            if args.input_weight_bits == 8:
                return tf.cast(img, tf.uint8), tf.cast(lab, tf.uint8)
            elif args.input_weight_bits == 4:
                return tf.cast(img, tf.uint4), tf.cast(lab, tf.uint4)
            else:
                raise ValueError("either 8 or 4 as input_weights_bits")

        train_ds = train_ds.map(norm)
        val_ds = val_ds.map(norm)
        test_ds = test_ds.map(norm)

        model_quantized = quantize(model, qparams=qparams)
        
        model_quantized.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=['accuracy'])

        #Quantized Aware Training
        hist_qt = model_quantized.fit(train_ds, 
                validation_data=val_ds, 
                epochs=args.epochs, 
                batch_size=64, 
                verbose=2)
        

        # Training metrics
        #train_loss = hist_qt.history['loss']
        #train_acc = hist_qt.history['accuracy']

        # Validation metrics
        val_loss = hist_qt.history['val_loss']
        #val_acc = hist_qt.history['val_accuracy']

        for i, v_l in enumerate(val_loss):
            print(f"epochs={i}")
            print(f"val_loss={v_l}:", flush=True)


        print(f"epochs={args.epochs}", flush=True)
        print(f"input_weight_bits={args.input_weight_bits}", flush=True)
        print(f"weight_bits={args.weight_bits}", flush=True)
        print(f"activation_bits={args.activation_bits}", flush=True)
        



if __name__ == "__main__":
    main()