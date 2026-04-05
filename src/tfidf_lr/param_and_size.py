import os
import numpy as np
import torch

def count_params(wrapper):
    """
    Count total parameters of a TF-IDF + LR ABSA wrapper.
    Returns total params and per-aspect breakdown.
    """
    params_per_aspect = {}
    total_params = 0

    for j, asp in enumerate(wrapper.aspects):
        # Mention detection
        mention_params = wrapper.mention_model.estimators_[j].coef_.size \
                         + wrapper.mention_model.estimators_[j].intercept_.size

        # Sentiment classifier
        sent_model = wrapper.sent_models[asp]["model"]
        if sent_model is not None:
            sent_params = sent_model.coef_.size + sent_model.intercept_.size
        else:
            sent_params = 0

        aspect_total = mention_params + sent_params
        params_per_aspect[asp] = aspect_total
        total_params += aspect_total

    return total_params, params_per_aspect

def get_model_size(path):
    """Returns Size of the model"""
    return os.path.getsize(path) / (1024**2)