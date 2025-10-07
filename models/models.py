import tensorflow as tf


def roshambo_net(input_shape=(64, 64, 1), num_classes=4):
    return tf.keras.Sequential([
        #tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),

        tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.ReLU(max_value=6.0),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
        
        tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.ReLU(max_value=6.0),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
        
        tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.ReLU(max_value=6.0),
        
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(24),
        tf.keras.layers.ReLU(max_value=6.0),

        tf.keras.layers.Dense(num_classes)
    ], name="Roshambo_ReLU6")
