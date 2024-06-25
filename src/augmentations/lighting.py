#author: Vít Tlustoš

import numpy as np

def adjust_brightnes(image, B=0):
    return np.clip(
        image + B,                              # I(x, y) + B
        0, 255
    ).astype(np.uint8)

def adjust_contrast(image, f=1.0):
    image = image.astype(np.float32)
    return np.clip(
       (image - 128) * f + 128,                 # f * (I(x, y) - 128) + 128
        0, 255
    ).astype(np.uint8)
