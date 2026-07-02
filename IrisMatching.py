import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize


def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = normalize(a, norm='l2')
    b = normalize(b, norm='l2')
    return a @ b.T


def _pairwise_l1_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=2)


def _pairwise_l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))


class IrisMatcher:
    """
    Implements the project-required iris matching pipeline:
    1) Fisher Linear Discriminant (via LDA) for dimension reduction
    2) nearest-center classifier for identification
    3) similarity-score generation for verification / ROC

    Expected input:
    - X_train: feature matrix from session 1 images, shape (n_train, d)
    - y_train: eye labels for training images, shape (n_train,)
    - X_test : feature matrix from session 2 images, shape (n_test, d)
    - y_test : eye labels for test images, shape (n_test,)
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.lda = None
        self.class_centers_ = None
        self.class_labels_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)

        n_classes = len(np.unique(y_train))
        max_components = max(1, n_classes - 1)
        n_components = self.n_components if self.n_components is not None else max_components
        n_components = min(n_components, max_components, X_train.shape[1])

        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_lda = self.lda.fit_transform(X_train, y_train)

        labels = np.unique(y_train)
        centers = []
        for label in labels:
            centers.append(X_train_lda[y_train == label].mean(axis=0))

        self.class_labels_ = labels
        self.class_centers_ = np.vstack(centers)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lda is None:
            raise ValueError("Call fit() before transform().")
        return self.lda.transform(np.asarray(X, dtype=float))

    def predict(self, X_test: np.ndarray, metric: str = "l2") -> np.ndarray:
        X_test_lda = self.transform(X_test)
        if metric == "l1":
            dist = _pairwise_l1_distance(X_test_lda, self.class_centers_)
            idx = np.argmin(dist, axis=1)
        elif metric == "l2":
            dist = _pairwise_l2_distance(X_test_lda, self.class_centers_)
            idx = np.argmin(dist, axis=1)
        elif metric == "cosine":
            sim = _safe_cosine_similarity(X_test_lda, self.class_centers_)
            idx = np.argmax(sim, axis=1)
        else:
            raise ValueError("metric must be one of: 'l1', 'l2', 'cosine'")
        return self.class_labels_[idx]

    def identification_crr(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_test = np.asarray(y_test)
        results = {}
        for metric in ["l1", "l2", "cosine"]:
            pred = self.predict(X_test, metric=metric)
            results[metric] = float(np.mean(pred == y_test))
        return results

    def verification_scores(self, X_test: np.ndarray, y_test: np.ndarray, metric: str = "cosine"):
        """
        Produce genuine / impostor scores for ROC.
        For each test sample, compare against all class centers.
        Positive pairs: sample vs its true class center.
        Negative pairs: sample vs all other class centers.
        """
        X_test_lda = self.transform(X_test)
        y_test = np.asarray(y_test)

        if metric == "cosine":
            score_matrix = _safe_cosine_similarity(X_test_lda, self.class_centers_)
            higher_is_better = True
        elif metric == "l1":
            score_matrix = -_pairwise_l1_distance(X_test_lda, self.class_centers_)
            higher_is_better = True
        elif metric == "l2":
            score_matrix = -_pairwise_l2_distance(X_test_lda, self.class_centers_)
            higher_is_better = True
        else:
            raise ValueError("metric must be one of: 'l1', 'l2', 'cosine'")

        binary_labels = []
        scores = []
        for i, true_label in enumerate(y_test):
            for j, center_label in enumerate(self.class_labels_):
                binary_labels.append(1 if center_label == true_label else 0)
                scores.append(score_matrix[i, j])

        return np.array(binary_labels), np.array(scores), higher_is_better

    def roc_metrics(self, X_test: np.ndarray, y_test: np.ndarray, metric: str = "cosine") -> dict:
        labels, scores, _ = self.verification_scores(X_test, y_test, metric=metric)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(roc_auc),
            "n_positive_pairs": int(labels.sum()),
            "n_negative_pairs": int((labels == 0).sum()),
        }


def run_iris_matching(X_train, y_train, X_test, y_test, n_components=None):
    """
    One-call helper for project scripts.
    """
    matcher = IrisMatcher(n_components=n_components)
    matcher.fit(X_train, y_train)

    crr = matcher.identification_crr(X_test, y_test)
    roc_info = {
        "l1": matcher.roc_metrics(X_test, y_test, metric="l1"),
        "l2": matcher.roc_metrics(X_test, y_test, metric="l2"),
        "cosine": matcher.roc_metrics(X_test, y_test, metric="cosine"),
    }
    return matcher, crr, roc_info
