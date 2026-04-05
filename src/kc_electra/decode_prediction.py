import pandas as pd
import numpy as np
from .helper_utils import _sigmoid, _unpack_logits

def decode_predictions(
    packed_logits: np.ndarray,
    aspects: list[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Convert raw packed model logits to a final label DataFrame.

    Decoding steps
    --------------
    1. Split packed logits into mention and sentiment logits.
    2. Apply sigmoid + threshold to decide whether each aspect is mentioned.
    3. For mentioned aspects, take argmax over sentiment logits and map 0 to 1,
       1 to 2 (negative / positive).
    4. Set non-mentioned aspects to 0.

    Parameters
    ----------
    packed_logits : (N, 3A) numpy array
    aspects       : list of aspect names (length A)
    thresholds    : dict mapping aspect name to mention probability threshold

    Returns
    -------
    pd.DataFrame of shape (N, A) with integer labels in {0, 1, 2}
    """
    num_aspects = len(aspects)
    mention_logits, sentiment_logits = _unpack_logits(packed_logits, num_aspects)

    mention_probs = _sigmoid(mention_logits)                    # (N, A)
    sent_bin      = np.argmax(sentiment_logits, axis=-1)        # 0 or 1
    sent_label    = sent_bin + 1                                # 1 or 2

    output = np.zeros_like(sent_label, dtype=int)
    for j, aspect in enumerate(aspects):
        mentioned      = (mention_probs[:, j] >= thresholds[aspect]).astype(int)
        output[:, j]   = np.where(mentioned, sent_label[:, j], 0)

    return pd.DataFrame(output, columns=aspects)
