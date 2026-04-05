import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def confusion_analysis(y_pred, y_true, aspects, labels, label_names):
    # Per-aspect confusion matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, asp in enumerate(aspects):
        yt = y_true[asp].to_numpy()
        yp = y_pred[asp].to_numpy()
        cm = confusion_matrix(yt, yp, labels=labels)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=axes[idx]
        )
        axes[idx].set_title(asp)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.suptitle("Confusion Matrix Per Aspects")
    plt.tight_layout()
    plt.show()

    # Overall confusion matrix (flattened)
    yt_all = y_true.values.ravel()
    yp_all = y_pred.values.ravel()
    cm_all = confusion_matrix(yt_all, yp_all, labels=labels)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm_all, annot=True, fmt="d", cmap="Oranges",
        xticklabels=label_names, yticklabels=label_names
    )
    plt.title("Overall Confusion Matrix (Flattened)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    
