# Main File

# Packages Needed:
# - cv2, numpy, glob, os, matplotlib, scikit-learn

# File Calls:
# 1. ImageEnhancement.py
# 2. IrisLocalization.py
# 3. IrisNormalization.py
# 4. FeatureExtraction.py
# 5. IrisMatching.py

################################################################################################################

# ------------Pre-processing Images (Run Once)------------
# Bei-Bei Xan

import os
import cv2
from glob import glob

from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhance_iris

###
# Helper function 
def preprocess_iris(img, M=64, N=512):
    loc = localize_iris(img)
    norm = normalize_iris(img, loc["pupil_circle"], loc["iris_circle"], M=M, N=N)
    enh, bg, corr = enhance_iris(norm)

    return {
        "localization": loc,
        "normalized": norm,
        "enhanced": enh,
        "background": bg,
        "corrected": corr
    }

# README: Following the .rar extract we were provided, have the datasets folder in the same directory as this file
INPUT_ROOT = "datasets/CASIA Iris Image Database (version 1.0)"

OUTPUT_ROOT = "pre_processed_dataset"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# Search all subfolders for bmps
bmp_files = glob(os.path.join(INPUT_ROOT, "**", "*.bmp"), recursive=True)

print(f"Found {len(bmp_files)} images")

for img_path in bmp_files:
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Run pre-processing pipeline - Main major call, see preprocess_iris function for details
    out = preprocess_iris(img)

    enhanced = out["enhanced"]

    # Mirror the file structure of original dataset in output folder
    relative_path = os.path.relpath(img_path, INPUT_ROOT)

    save_path = os.path.join(OUTPUT_ROOT, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(save_path, enhanced)
###

print("\nImages Pre-Processed")


################################################################################################################

# ------------Feature Extraction------------
# Nick Weda

from FeatureExtraction import build_train_test_from_pre_processed_dataset

X_train, y_train, X_test, y_test = build_train_test_from_pre_processed_dataset()

print("\nFeatures Extracted and Train/Test Split built")


################################################################################################################

# ------------Similarity Matching------------
# Jigang Xie

from IrisMatching import run_iris_matching
import matplotlib.pyplot as plt

matcher, crr, roc_info = run_iris_matching(
    X_train,
    y_train,
    X_test,
    y_test,
    n_components=200
)

print("\nIris Matcher Ran")



################################################################################################################

# ------------Performance Evaluation ------------

from PerformanceEvaluation import evaluate_and_plot

evaluate_and_plot(crr, roc_info)