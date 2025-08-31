from __future__ import annotations
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam




def build_unet(input_ch: int, sz: int = 32, base: int = 32) -> Model:
    inp = layers.Input(shape=(sz, sz, input_ch))
    # enc1
    x1 = layers.Conv2D(base, 3, padding="same", activation="relu")(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv2D(base, 3, padding="same", activation="relu")(x1)
    x1 = layers.BatchNormalization()(x1)
    p1 = layers.MaxPooling2D(2)(x1)
    # enc2
    x2 = layers.Conv2D(base*2, 3, padding="same", activation="relu")(p1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv2D(base*2, 3, padding="same", activation="relu")(x2)
    x2 = layers.BatchNormalization()(x2)
    p2 = layers.MaxPooling2D(2)(x2)
    # bottleneck
    b = layers.Conv2D(base*4, 3, padding="same", activation="relu")(p2)
    b = layers.BatchNormalization()(b)
    b = layers.Conv2D(base*4, 3, padding="same", activation="relu")(b)
    b = layers.BatchNormalization()(b)
    # dec1
    u1 = layers.Concatenate()([layers.Conv2DTranspose(base*2, 2, strides=2, padding="same")(b), x2])
    d1 = layers.Conv2D(base*2, 3, padding="same", activation="relu")(u1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Conv2D(base*2, 3, padding="same", activation="relu")(d1)
    d1 = layers.BatchNormalization()(d1)
    # dec2
    u2 = layers.Concatenate()([layers.Conv2DTranspose(base, 2, strides=2, padding="same")(d1), x1])
    d2 = layers.Conv2D(base, 3, padding="same", activation="relu")(u2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Conv2D(base, 3, padding="same", activation="relu")(d2)
    d2 = layers.BatchNormalization()(d2)
    out = layers.Conv2D(1, 1, activation="sigmoid")(d2)
    return Model(inputs=inp, outputs=out)