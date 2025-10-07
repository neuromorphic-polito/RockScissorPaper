from datetime import timedelta
import cv2 as cv
import dv_processing as dv
import numpy as np
import os



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
        accumulator.accept(events)
        frame = accumulator.generateFrame()
        img = cv.resize(frame.image, dsize=resize_shape)
        img = np.expand_dims(img, axis=-1)

        cv.imshow("Preview", frame.image)
        cv.waitKey(2)

        img_ls.append(img) #(64,64,1)

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



#/media/cass/SanDisk/dvs_recording_lab/dv_scissor_cass.aedat4
def save_images_to_folder(images, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        class_dir = os.path.join(out_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{i}.png")
        #Image.fromarray(img.squeeze(), mode="L").save(img_path)
        cv.imwrite(img_path, img.squeeze())


