# Iris Recognition via Ma et al. (2003)

Classical iris recognition pipeline (localization, normalization, enhancement, Gabor feature extraction, LDA-based matching) implemented from scratch on the CASIA-IrisV1 dataset, built for Columbia's COMS 5293 Applied Machine Learning for Computer Vision.

No deep learning. No pretrained models. Just coarse-to-fine boundary detection, polar coordinate remapping, and a multi-channel filter bank, reimplementing the design from Ma, Wang, and Tan's *Personal Identification Based on Iris Texture Analysis*.

> **Group project.** Team: Nicholas Rodriguez Weda, Beibei Xian, Jigang Xie.
> **My contribution:** the preprocessing pipeline — `IrisLocalization.py`, `IrisNormalization.py`, `ImageEnhancement.py` (coarse-to-fine pupil/iris boundary detection, Cartesian-to-polar unwrapping, CLAHE-based illumination correction).

## Pipeline Overview

```
Original grayscale iris image
        │
        ▼
Iris Localization (pupil + outer iris boundary)
        │
        ▼
Iris Normalization (Cartesian → polar, 64 x 512 strip)
        │
        ▼
Image Enhancement (illumination correction + CLAHE)
        │
        ▼
Feature Extraction (Gabor-like filter bank, block statistics)
        │
        ▼
Iris Matching (LDA dimensionality reduction + nearest-center classification)
        │
        ▼
Performance Evaluation (CRR, ROC)
```

## Dataset

CASIA-IrisV1 — 108 eyes, 7 images per eye across two sessions (320x280 BMP). Session 1 (3 images/eye) is used for training, Session 2 (4 images/eye) for testing, matching the paper's split.

## Module Breakdown

### 1. `IrisLocalization.py`
Detects the pupil and outer iris boundary using a coarse-to-fine approach: a rough pupil center is estimated from horizontal/vertical intensity projection profiles (the pupil is the darkest region in the image), refined within a 120x120 crop via thresholding and centroid detection, then exact pupil and iris circles are found with edge detection + Hough circle detection in a guided search region.

| Parameter | Role |
|---|---|
| `window = 120` | local crop size for pupil refinement |
| `pupil_circle (xp, yp, rp)` | detected pupil center + radius |
| `iris_circle (xi, yi, ri)` | detected iris center + radius |
| Hough params (`dp`, `minDist`, `param1`, `param2`, `minRadius`, `maxRadius`) | tuned per search region |

### 2. `IrisNormalization.py`
Unwraps the iris ring from Cartesian to polar coordinates, producing a fixed-size 64x512 rectangular strip. Each angular position samples along the radial line between the pupil and iris boundary at that angle, reducing deformation from pupil size changes and imaging differences.

| Parameter | Role |
|---|---|
| `M = 64` | radial samples |
| `N = 512` | angular samples |

### 3. `ImageEnhancement.py`
Improves the normalized strip by correcting uneven illumination and boosting local texture contrast. A coarse background is estimated from 16x16 block means, upsampled via bicubic interpolation, and subtracted; CLAHE is then applied as a practical stand-in for the paper's region-by-region local histogram equalization.

| Parameter | Role |
|---|---|
| `block_size = 16` | illumination estimation block size |
| `clipLimit = 2.0` | CLAHE contrast limit |
| `tileGridSize = (8, 8)` | CLAHE tile layout |

### 4. `FeatureExtraction.py`
Applies a two-channel Gabor-like filter bank to the top ROI of the enhanced image (`roi_rows = 48`), divides the filtered response into 8x8 blocks, and computes mean/standard deviation per block as the feature vector.

### 5. `IrisMatching.py`
Projects high-dimensional feature vectors into a lower-dimensional space via Fisher Linear Discriminant Analysis (`n_components = 200`, empirically best), then classifies with a nearest-center rule under L1, L2, and cosine similarity.

### 6. `PerformanceEvaluation.py`
Computes Correct Recognition Rate (CRR) and ROC curves from `IrisMatching`'s output.

### 7. `IrisRecognition.py`
Main script wiring all of the above into a single end-to-end pipeline.

## Results

**Preprocessing** — stable, accurate pupil localization across tested samples; visually reasonable outer iris boundaries; consistent 64x512 normalized strips; enhancement measurably improves texture visibility and local contrast.

**Feature extraction** — uniform-length feature vectors across all images; vectors cluster by eye identity and separate across different eyes, with block statistics capturing meaningful inter-image variation.

**Matching** — clear discriminative signal between classes; cosine similarity is the strongest of the three metrics tested; LDA reduces dimensionality while preserving class separability; results are consistent across repeated runs.

**Overall CRR:** 60-70%, versus the paper's reported 80%+.

## Where the Pipeline Falls Short

- **No eyelid/eyelash/reflection masking** — these artifacts remain visible through normalization and enhancement and likely suppress downstream matching accuracy
- **Circular boundary assumption** — the paper notes real pupil/iris boundaries are usually not perfectly concentric; this implementation assumes they are
- **CLAHE as an approximation** — substitutes for the paper's exact region-by-region local histogram equalization rather than reproducing it directly
- **Unspecified filter parameters** — the paper doesn't publish its Gabor filter frequency values, so ours were tuned empirically rather than matched exactly
- **Only 2 filter channels** — may not capture the full range of texture variation the paper's design intends
- **Simplified matching** — limited per-class training samples and only approximate rotation compensation, likely the largest single gap versus the paper's reported CRR

## What We'd Do Next

- Add eyelid, eyelash, and specular reflection segmentation before feature extraction
- Tune Hough circle parameters more systematically and add stronger boundary constraints for difficult images
- Implement the paper's exact local histogram equalization instead of the CLAHE approximation
- Cross-validate Gabor filter frequencies instead of setting them empirically
- Add proper rotation compensation and explore alternative dimensionality reduction techniques
- Add a quality gate to reject severely occluded or low-quality iris images before they reach matching

## Run It

```bash
python IrisRecognition.py
```

Reads CASIA-IrisV1 images through the full pipeline and outputs CRR and ROC results via `PerformanceEvaluation.py`.

## Reference

Ma, L., Wang, Y., & Tan, T. (2003). *Personal Identification Based on Iris Texture Analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence.

---
Team: Nicholas Rodriguez Weda · Beibei Xian · Jigang Xie · COMS 5293, Columbia University
