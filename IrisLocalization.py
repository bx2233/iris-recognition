#From Notebook
# writefile /content/drive/MyDrive/5293_IrisProject/IrisLocalization.py
import cv2
import numpy as np


def rough_pupil_center(img):
    """
    Estimate coarse pupil center from horizontal/vertical projection minima.
    Input:
        img: grayscale iris image, shape (H, W)
    Output:
        (x0, y0): rough pupil center
    """
    # darkest row/column gives a rough pupil center
    row_profile = img.mean(axis=1)
    col_profile = img.mean(axis=0)

    y0 = int(np.argmin(row_profile))
    x0 = int(np.argmin(col_profile))
    return x0, y0


def refine_pupil_center(img, x0, y0, window=120):
    """
    Refine pupil center inside a local crop around rough center.
    Uses thresholding + centroid of largest dark connected component.
    """
    h, w = img.shape
    half = window // 2

    # crop around rough center
    x1 = max(0, x0 - half)
    x2 = min(w, x0 + half)
    y1 = max(0, y0 - half)
    y2 = min(h, y0 + half)

    crop = img[y1:y2, x1:x2]

    blur = cv2.GaussianBlur(crop, (5, 5), 0)

    # isolate dark pupil region
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    if num_labels <= 1:
        return x0, y0, 30

    # use largest dark component
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx_local, cy_local = centroids[largest_idx]
    area = stats[largest_idx, cv2.CC_STAT_AREA]

    x_ref = int(x1 + cx_local)
    y_ref = int(y1 + cy_local)
    r_est = int(np.sqrt(area / np.pi))

    return x_ref, y_ref, max(r_est, 10)


def detect_circles(img, x_hint, y_hint, pupil_r_est):
    """
    Detect pupil and iris circles using HoughCircles.
    Returns:
        pupil_circle = (xp, yp, rp)
        iris_circle  = (xi, yi, ri)
    """
    blur = cv2.GaussianBlur(img, (7, 7), 1.5)

    pupil_circle = None
    iris_circle = None

    # detect pupil circle first
    pupil_candidates = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=20,
        minRadius=max(10, pupil_r_est - 15),
        maxRadius=pupil_r_est + 20
    )

    if pupil_candidates is not None:
        pupil_candidates = np.round(pupil_candidates[0]).astype(int)

        best = None
        best_dist = 1e9
        for c in pupil_candidates:
            x, y, r = c
            dist = (x - x_hint) ** 2 + (y - y_hint) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (x, y, r)
        pupil_circle = best
    else:
        pupil_circle = (x_hint, y_hint, pupil_r_est)

    xp, yp, rp = pupil_circle

    # detect outer iris circle near pupil center
    iris_candidates = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=30,
        minRadius=max(rp + 40, 70),
        maxRadius=140
    )

    if iris_candidates is not None:
        iris_candidates = np.round(iris_candidates[0]).astype(int)

        best = None
        best_dist = 1e9
        for c in iris_candidates:
            x, y, r = c
            dist = (x - xp) ** 2 + (y - yp) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (x, y, r)
        iris_circle = best
    else:
        iris_circle = (xp, yp, rp + 60)

    return pupil_circle, iris_circle


def draw_circles(img, pupil_circle, iris_circle):
    """
    Draw detected circles for debugging.
    """
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    xp, yp, rp = pupil_circle
    xi, yi, ri = iris_circle

    # green = pupil, blue = iris
    cv2.circle(out, (xp, yp), rp, (0, 255, 0), 2)
    cv2.circle(out, (xi, yi), ri, (255, 0, 0), 2)
    cv2.circle(out, (xp, yp), 2, (0, 0, 255), -1)
    cv2.circle(out, (xi, yi), 2, (0, 255, 255), -1)

    return out


def localize_iris(img):
    """
    Full iris localization pipeline.
    Input:
        img: grayscale image
    Output:
        result dict with pupil/iris circles and debug image
    """
    # full pipeline: rough center -> refine -> circles -> overlay
    x0, y0 = rough_pupil_center(img)
    x1, y1, r_est = refine_pupil_center(img, x0, y0, window=120)
    pupil_circle, iris_circle = detect_circles(img, x1, y1, r_est)
    overlay = draw_circles(img, pupil_circle, iris_circle)

    return {
        "rough_center": (x0, y0),
        "refined_center": (x1, y1),
        "pupil_circle": pupil_circle,
        "iris_circle": iris_circle,
        "overlay": overlay,
    }