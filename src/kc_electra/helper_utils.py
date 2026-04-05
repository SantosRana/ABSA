import numpy as np
from transformers import Trainer

class ContiguousTrainer(Trainer):
    """Trainer that forces parameter tensors to be contiguous before checkpointing."""

    def _save_checkpoint(self, model, trial, **kwargs):
        """Ensure parameter storage layout is contiguous, then save normally."""
        # Some serialization paths can fail or slow down with non-contiguous tensors.
        for name, param in model.named_parameters():
            if not param.data.is_contiguous():
                # Materialize a contiguous copy to keep checkpoint writes stable.
                param.data = param.data.contiguous()

        # Delegate to the default Trainer checkpoint implementation.
        super()._save_checkpoint(model, trial, **kwargs)


def _unpack_logits(packed: np.ndarray, num_aspects: int) -> tuple:
    """
    Split the packed model output into mention and sentiment logit arrays.

    Parameters
    ----------
    packed      : (N, 3A) numpy array from Trainer.predict
    num_aspects : int, number of aspects A

    Returns
    -------
    mention_logits   : (N, A)    pre-sigmoid scores for mention detection
    sentiment_logits : (N, A, 2) pre-softmax scores for sentiment
    """
    packed           = np.asarray(packed)
    mention_logits   = packed[:, :num_aspects]                              # (N, A)
    sentiment_logits = packed[:, num_aspects:].reshape(-1, num_aspects, 2) # (N, A, 2)
    return mention_logits, sentiment_logits


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid (numerically stable)."""
    return 1.0 / (1.0 + np.exp(-x))


def _tokenize_batch(batch: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenisation map function for HuggingFace Dataset.map().

    Parameters
    ----------
    batch      : dict with key "text" (list of strings)
    tokenizer  : HuggingFace tokenizer
    max_length : int, sequence length after truncation / padding

    Returns
    -------
    dict containing "input_ids" and "attention_mask"
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


