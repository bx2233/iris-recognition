## From Notebook
# writefile /content/drive/MyDrive/5293_IrisProject/IrisNormalization.py
import numpy as np
import cv2


def normalize_iris(img, pupil_circle, iris_circle, M=64, N=512):
    """
    Normalize iris from Cartesian coordinates to polar coordinates.

    Input:
        img: grayscale image, shape (H, W)
        pupil_circle: (xp, yp, rp)
        iris_circle: (xi, yi, ri)
        M: number of radial samples
        N: number of angular samples

    Output:
        norm_img: normalized iris image of shape (M, N)
    """
    xp, yp, rp = pupil_circle
    xi, yi, ri = iris_circle

    # sample full 360 degrees
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    norm_img = np.zeros((M, N), dtype=np.uint8)

    for j, ang in enumerate(theta):
        # point on pupil boundary
        x_p = xp + rp * np.cos(ang)
        y_p = yp + rp * np.sin(ang)

        # point on iris boundary
        x_i = xi + ri * np.cos(ang)
        y_i = yi + ri * np.sin(ang)

        for i in range(M):
            # interpolate from inner to outer boundary
            r = i / (M - 1)
            x = (1 - r) * x_p + r * x_i
            y = (1 - r) * y_p + r * y_i

            # sample image intensity at interpolated point
            value = cv2.getRectSubPix(img, (1, 1), (float(x), float(y)))[0, 0]
            norm_img[i, j] = value

    return norm_img