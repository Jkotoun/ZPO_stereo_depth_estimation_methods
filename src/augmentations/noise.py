#author: Vít Tlustoš

import numpy as np

def add_gaussian_noise(img, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, img.shape)                # N(mean, sigma^2)
    return np.clip(img + noise * 255, 0, 255).astype(np.uint8)      # I(x, y) + N(mean, sigma^2) * 255


def add_salt_and_pepper_noise(img, salt_pepper_ratio=0.5, amount=0.05):
    num_invalid_pixels = int(amount * img.shape[0] * img.shape[0])
    # 1. salt noise
    coords = [
        np.random.randint(0, i - 1, num_invalid_pixels) 
        for i in img.shape
    ]
    img[coords[0], coords[1]] = 255
    # 2. pepper noise
    num_pepper = int(num_invalid_pixels * salt_pepper_ratio)
    coords = [
        np.random.randint(0, i - 1, num_pepper)
        for i in img.shape
    ]
    img[coords[0], coords[1]] = 0
    return img
