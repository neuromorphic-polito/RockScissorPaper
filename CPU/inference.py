import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import timedelta
import cv2 as cv
import dv_processing as dv
import numpy as np
import tensorflow as tf
from models.models import roshambo_net
from collections import deque, Counter

# keep last 5 predictions
PRED_HISTORY = deque(maxlen=5)


#Uses categories {paper: 0, rock: 2, scissors: 1, background: 3} for 


CATEGORIES = {
    0: "PAPER",
    1: "SCISSORS",
    2: "ROCK",
    3: "BACKGROUND"
}

# load reference images once
CATEGORY_IMAGES = {
    "PAPER": cv.imread("../imgs/paper.png"),
    "SCISSORS": cv.imread("../imgs/scissors.png"),
    "ROCK": cv.imread("../imgs/rock.png"),
    "BACKGROUND": cv.imread("../imgs/bg.jpg")
}




def category(prediction):
    label = CATEGORIES.get(prediction)
    return label




model = roshambo_net()
model.load_weights("./ckpt/same_epoch_40.keras")
model.trainable = False
model.compile(metrics=['accuracy'])
model.summary()
print("model loaded")




def aedat4_to_images(filename: str,
                     acculumation_by: str = "events",
                     acc_value: int = 4000,
                     resize_shape: tuple = (64, 64)) -> list:
    """
    Convert an AEDAT4 event file into a list of images using dv-processing.

    Args:
        filename (str): Path to the AEDAT4 file.
        acculumation_by (str): 'events' or 'milliseconds' for slicing.
        acc_value (int): Number of events or milliseconds for accumulation.
        resize_shape (tuple): (width, height) to resize output frames.

    Returns:
        list: List of accumulated frames as numpy arrays.
    """

    img_ls = []

    def slicing_callback(events: dv.EventStore):
        global model
        
        accumulator.accept(events)
        frame = accumulator.generateFrame()
        img = cv.resize(frame.image, dsize=resize_shape)

        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

            
        # predict class
        pred_idx = int(tf.argmax(model(img.astype("uint8")), axis=1))

        
        # add to history
        PRED_HISTORY.append(pred_idx)

            # majority voting
        if len(PRED_HISTORY) > 0:
            pred_label = Counter(PRED_HISTORY).most_common(1)[0][0]


        # camera frame -> make 3 channels
        cam_img = cv.cvtColor(frame.image, cv.COLOR_GRAY2BGR)


        if pred_label == 0:
            pred_img = CATEGORY_IMAGES['SCISSORS']
        if pred_label == 1:
            pred_img = CATEGORY_IMAGES['ROCK']
        if pred_label == 2:
            pred_img = CATEGORY_IMAGES['PAPER']
        if pred_label == 3:
            pred_img = CATEGORY_IMAGES['BACKGROUND']


        pred_img = cv.resize(pred_img, (cam_img.shape[1], cam_img.shape[0]))

        # concatenate side by side
        combined = np.hstack((cam_img, pred_img))

        # show in single window
        cv.imshow("Camera + Prediction", combined)

        if cv.waitKey(1) & 0xFF == ord('q'):
            capture.stop()



    # Open recording or camera
    if filename == "camera":
        capture = dv.io.camera.open()
    else:
        capture = dv.io.MonoCameraRecording(filename)

    if not capture.isEventStreamAvailable():
        raise RuntimeError("Input does not provide an event stream.")

    # Initialize slicer
    slicer = dv.EventStreamSlicer()
    if acculumation_by == "milliseconds":
        slicer.doEveryTimeInterval(timedelta(milliseconds=acc_value), slicing_callback)
    elif acculumation_by == "events":
        slicer.doEveryNumberOfElements(acc_value, slicing_callback)
    else:
        raise ValueError("Invalid acculumation_by. Use 'events' or 'milliseconds'.")

    # Initialize an accumulator with some resolution
    accumulator = dv.Accumulator(capture.getEventResolution())
    accumulator.setMinPotential(0.0)
    accumulator.setMaxPotential(2.0)
    accumulator.setNeutralPotential(0)
    accumulator.setEventContribution(1)
    accumulator.setDecayFunction(dv.Accumulator.Decay.STEP)
    accumulator.setIgnorePolarity(True)
    accumulator.setSynchronousDecay(False)

    cv.namedWindow("Preview", cv.WINDOW_NORMAL)

    # Process the AEDAT4 file
    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

    return img_ls






recording = aedat4_to_images(filename="camera", acc_value=40000)
