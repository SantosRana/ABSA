import numpy as np
from sklearn.metrics import f1_score
from .helper_utils import _sigmoid, _unpack_logits

def compute_metrics_fn(eval_pred, num_aspects: int) -> dict:
    """
    Compute end-to-end 3-class macro F1 for HuggingFace Trainer.

    This function is passed as `compute_metrics` to `Trainer`.  It operates on
    the packed logits and integer labels produced during evaluation.

    Parameters
    ----------
    eval_pred   : EvalPrediction namedtuple (logits, label_ids)
    num_aspects : number of aspects (A)

    Returns
    -------
    dict with key "e2e_macro_f1" (float, averaged across aspects)
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    mention_logits, sentiment_logits = _unpack_logits(logits, num_aspects)

    # Mention predictions using fixed 0.5 threshold (thresholds are tuned
    # only after training completes)
    mention_pred = (_sigmoid(mention_logits) >= 0.5).astype(int)

    # Sentiment: 0 â†’ label 1 (neg), 1 â†’ label 2 (pos)
    sent_label = np.argmax(sentiment_logits, axis=-1) + 1       # (N, A)

    # Combine: if not mentioned â†’ 0
    preds = np.where(mention_pred, sent_label, 0)               # (N, A)

    # Per-aspect macro F1, then average
    per_aspect_f1 = [
        f1_score(labels[:, j], preds[:, j], average="macro", zero_division=0)
        for j in range(num_aspects)
    ]
    return {"e2e_macro_f1": float(np.mean(per_aspect_f1))}