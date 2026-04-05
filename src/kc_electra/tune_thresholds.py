import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def tune_mention_thresholds(
    mention_probs: np.ndarray,
    y_val: pd.DataFrame,
    aspects: list[str],
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Find the per-aspect mention probability threshold that maximises F1 on the
    validation set.

    Parameters
    ----------
    mention_probs : (N, A) sigmoid probabilities from the mention head
    y_val         : validation label DataFrame (values 0/1/2)
    aspects       : list of aspect names matching the columns of y_val
    grid          : 1-D array of thresholds to try (default: 0.1 â€¦ 0.9)

    Returns
    -------
    dict mapping aspect name â†’ best threshold (float)
    """
    if grid is None:
        grid = np.arange(0.10, 0.91, 0.10)

    # Ground-truth: was the aspect mentioned at all?
    y_true_mention = (y_val[aspects].values != 0).astype(int)

    thresholds = {}
    for j, aspect in enumerate(aspects):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            y_pred = (mention_probs[:, j] >= t).astype(int)
            f1     = f1_score(y_true_mention[:, j], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[aspect] = best_t

    return thresholds