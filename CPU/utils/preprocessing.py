import tensorflow as tf

def norm(img,lab):
    return tf.cast(img, tf.uint8), tf.cast(lab, tf.uint8),
