import torch

def compute_class_weights(train_labels):
    """
    Compute class weights for a multi-label classification problem.

    Args:
        train_labels: (N, A) DataFrame or array with values {0,1,2}
                      - 0 = not mentioned
                      - 1 = negative sentiment
                      - 2 = positive sentiment

    Returns:
        mention_pos_weight: scalar weight for positive class in mention detection
        sentiment_class_weights: tensor of weights for sentiment classification [neg, pos]
    """

    # Convert labels to a torch tensor of type long
    y = torch.tensor(train_labels.values, dtype=torch.long)


    # Mention weight
    # -------------------------
    # Binary target: 1 if mentioned (labels != 0), else 0
    mention_target = (y != 0).float()

    # Count positives (mentioned) and negatives (not mentioned)
    num_positive = mention_target.sum()
    num_negative = mention_target.numel() - num_positive

    # Positive weight for mention detection (used in BCEWithLogitsLoss)
    mention_pos_weight = num_negative / (num_positive + 1e-8)

    # Sentiment weights
    # -------------------------
    # Only consider samples where mention_target == 1 (labels != 0)
    sent_labels = y[y != 0]

    # Count negative and positive sentiment labels
    num_neg = (sent_labels == 1).sum()
    num_pos = (sent_labels == 2).sum()

    # Total number of sentiment samples
    total = num_neg + num_pos

    # Compute class weights for sentiment classification
    # Formula: total / (2 * count_of_class)
    weight_neg = total / (2 * (num_neg + 1e-8))
    weight_pos = total / (2 * (num_pos + 1e-8))

    # Store weights in tensor [neg, pos]
    sentiment_class_weights = torch.tensor([weight_neg, weight_pos])

    return mention_pos_weight, sentiment_class_weights