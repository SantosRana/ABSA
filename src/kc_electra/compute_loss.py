import torch
import torch.nn as nn

def _compute_loss(
    mention_logits: torch.Tensor,
    sentiment_logits: torch.Tensor,
    labels: torch.Tensor,
    mention_pos_weight: torch.Tensor,
    sentiment_class_weights: torch.Tensor
) -> torch.Tensor:
    """
    Combined mention + sentiment loss with fixed dataset-level weights.

    Mention loss
    ------------
    Binary cross-entropy over all (sample, aspect) pairs.
    Target = 1 if label != 0, else 0.

    Sentiment loss
    --------------
    Cross-entropy only on (sample, aspect) pairs where the aspect IS mentioned
    (label != 0).  Labels 1 and 2 are remapped to 0 and 1 respectively.

    Parameters
    ----------
    mention_logits   : (B, A) raw mention scores
    sentiment_logits : (B, A, 2) raw sentiment scores
    labels           : (B, A) integer ground-truth labels {0, 1, 2}

    Returns
    -------
    torch.Tensor : scalar combined loss
    """
    y = labels.long()

    # -----------------------
    # Mention Loss
    # -----------------------
    mention_target = (y != 0).float()

    mention_loss = nn.BCEWithLogitsLoss(
        pos_weight=mention_pos_weight.to(mention_logits.device)
    )(mention_logits, mention_target)

    # -----------------------
    # Sentiment Loss
    # -----------------------
    B, A, _ = sentiment_logits.shape

    sent_target = (y - 1).clamp(0, 1)

    flat_logits = sentiment_logits.reshape(B * A, 2)
    flat_target = sent_target.reshape(B * A)
    mask = (y != 0).reshape(B * A)

    valid_logits = flat_logits[mask]
    valid_target = flat_target[mask]

    if valid_target.numel() > 0:
        sentiment_loss = nn.CrossEntropyLoss(
            weight=sentiment_class_weights.to(sentiment_logits.device)
        )(valid_logits, valid_target)
    else:
        sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device)

    return mention_loss + sentiment_loss
