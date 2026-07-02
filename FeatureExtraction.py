import os
from glob import glob
import cv2
import numpy as np


# Gabor Filter Implementation
def gabor_filter(dx, dy, f, kernel_size=31):
    # dx, dy - Spread of Gaussian
    # f - frequency of cosine
    # kernel_size - size of the filter

    half = kernel_size // 2 # floor to odd

    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32) # create x,y coords

    gaussian = (1.0 / (2.0 * np.pi * dx * dy)) * np.exp(
        -0.5 * ((x ** 2) / (dx ** 2) + (y ** 2) / (dy ** 2))
    ) # define the gaussian

    radial = np.sqrt(x ** 2 + y ** 2) # distance from center
    modulation = np.cos(2.0 * np.pi * f * radial) # Modultion Function, Cosine Wave

    kernel = gaussian * modulation # combine gaussian and cosine radial modulation
    kernel = kernel - kernel.mean() # normalize to zero mean

    norm = np.sqrt(np.sum(kernel ** 2)) # l2 norm for normalization
    if norm > 0:
        kernel = kernel / norm

    return kernel.astype(np.float32)


def extract_features_from_image(
    img,
    roi_rows=48, # Size of ROI as specified in paper
    block_size=8,
    channels=((3.0, 1.5, 0.1), (4.5, 1.5, 0.07)),
    kernel_size=31,
):
    # block_size - division of image into blocks of set size
    # channels - Tuple of (dx, dy, f) for each filter channel, 
        # dx and dy are given in the paper
        # f is a tuning parameter that is unspecified in the paper
        # frequency sweep results in 0.1 and 0.07 yielding the best CRR

    # Crop Image to ROI
    roi = img[:roi_rows, :].astype(np.float32)

    # Accumulate mean and standard deviation values for each block and channel
    extracted_image_features = []

    # For each filter channel
    for dx, dy, f in channels:
        kernel = gabor_filter(dx, dy, f, kernel_size=kernel_size)

        # Apply filter to ROI
        response = cv2.filter2D(
            roi,
            ddepth=cv2.CV_32F,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT
        )

        mag = np.abs(response) # magnitude
        h, w = mag.shape # shape

        # Loop over blocks, compute mean and std, and add to  list
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = mag[y:y + block_size, x:x + block_size]
                m = float(np.mean(block))
                sigma = float(np.mean(np.abs(block - m)))
                extracted_image_features.extend([m, sigma])

    return np.asarray(extracted_image_features, dtype=np.float32)


def build_train_test_from_pre_processed_dataset(input_root="pre_processed_dataset", image_ext="bmp"):
    image_files = glob(os.path.join(input_root, "**", f"*.{image_ext}"), recursive=True)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        feature_vector = extract_features_from_image(img)

        parts = os.path.normpath(img_path).split(os.sep)
        subject = parts[-3]   # eye ID
        session = parts[-2]   # "1" or "2"

        label = subject

        # Split based on session
        if session == "1":
            X_train.append(feature_vector)
            y_train.append(label)
        elif session == "2":
            X_test.append(feature_vector)
            y_test.append(label)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test

# if __name__ == "__main__":
#     X_train, y_train, X_test, y_test = build_train_test_from_pre_processed_dataset()

#     print("\nFirst training label:")
#     print(y_train[0])

#     print("\nFirst training feature vector:")
#     print(X_train[0])

#     print("\nFirst training feature vector shape:")
#     print(X_train[0].shape)