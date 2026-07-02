# From Notebook
# writefile /content/drive/MyDrive/5293_IrisProject/ImageEnhancement.py
import cv2
import numpy as np


def estimate_background(norm_img, block_size=16):
    """
    Estimate coarse illumination background using block means.
    """
    h, w = norm_img.shape
    bg_small = np.zeros((h // block_size, w // block_size), dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # average each local block
            block = norm_img[i:i+block_size, j:j+block_size]
            bg_small[i // block_size, j // block_size] = np.mean(block)

    # resize coarse background back to full image size
    bg_img = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return bg_img


def enhance_iris(norm_img):
    """
    Enhance normalized iris image by illumination correction
    and local histogram equalization.
    """
    norm_float = norm_img.astype(np.float32)

    # estimate and remove uneven background illumination
    bg_img = estimate_background(norm_img, block_size=16)
    corrected = norm_float - bg_img

    # rescale corrected image to 0-255
    corrected = corrected - corrected.min()
    if corrected.max() > 0:
        corrected = corrected / corrected.max() * 255.0

    corrected = corrected.astype(np.uint8)

    # improve local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    return enhanced, bg_img.astype(np.uint8), corrected