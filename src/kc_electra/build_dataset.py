import pandas as pd
import numpy as np
from datasets import Dataset
from .helper_utils import _tokenize_batch

def build_dataset(
    X: pd.Series,
    y_df: pd.DataFrame | None,
    tokenizer,
    aspects: list[str],
    max_length: int,
) -> Dataset:
    """
    Build a tokenised HuggingFace Dataset from raw texts and optional labels.

    Parameters
    ----------
    X          : iterable of raw text strings
    y_df       : DataFrame with one column per aspect (values 0/1/2).
                 Pass None for unlabelled inference data.
    tokenizer  : HuggingFace tokenizer
    aspects    : list of aspect column names
    max_length : maximum token sequence length

    Returns
    -------
    Dataset with columns: input_ids, attention_mask [, labels]
    """
    data = {"text": list(X)}
    if y_df is not None:
        data["labels"] = y_df[aspects].values.astype(np.int64)

    ds = Dataset.from_dict(data)

    # Tokenise in batches for efficiency
    ds = ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, max_length),
        batched=True,
    )

    # Select only the columns needed by the model
    columns = ["input_ids", "attention_mask"] + (["labels"] if y_df is not None else [])
    ds.set_format(type="torch", columns=columns)
    return ds