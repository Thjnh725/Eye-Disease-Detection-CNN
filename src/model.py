import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)

def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

def conv_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    # thêm SE nhẹ
    x = se_block(x)

    return x

def create_model_v2():
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Block 1
    x = conv_block(x, 32)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = conv_block(x, 64)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = conv_block(x, 128)
    x = layers.MaxPooling2D(2)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x) 
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(4, activation='softmax')(x)

    return keras.Model(inputs, outputs)