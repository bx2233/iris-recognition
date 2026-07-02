import matplotlib.pyplot as plt


def evaluate_and_plot(crr, roc_info):
    print("\nResults:")

    print("\nIdentification CRR:")
    for metric, val in crr.items():
        print(f"{metric}: {val:.4f}")

    print("\nROC AUC:")
    print("L1    :", roc_info["l1"]["auc"])
    print("L2    :", roc_info["l2"]["auc"])
    print("Cosine:", roc_info["cosine"]["auc"])

    # Plot ROC (cosine)
    plt.figure(figsize=(6, 5))
    plt.plot(
        roc_info["cosine"]["fpr"],
        roc_info["cosine"]["tpr"],
        label=f"Cosine ROC (AUC = {roc_info['cosine']['auc']:.4f})"
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Verification ROC Curve (Cosine Similarity)")
    plt.legend()
    plt.tight_layout()
    plt.show()