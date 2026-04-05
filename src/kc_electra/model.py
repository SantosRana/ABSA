import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from .tune_thresholds import tune_mention_thresholds
from .build_dataset import build_dataset
from .class_weights import compute_class_weights
from .compute_loss import _compute_loss
from .compute_metrics import compute_metrics_fn
from .helper_utils import _sigmoid, _unpack_logits, ContiguousTrainer
from .decode_prediction import decode_predictions
from pathlib import Path


"""
Aspect-Based Sentiment Analysis (ABSA) using KcELECTRA (Korean)
================================================================

This module implements a shared-encoder multi-task model for
Aspect-Based Sentiment Analysis (ABSA).

Each aspect (FOOD, PRICE, SERVICE, AMBIENCE) is modeled using:

1) Mention detection head
   - Binary classification
   - Predicts whether the aspect is mentioned

2) Sentiment classification head
   - Binary classification
   - Predicts negative or positive sentiment
   - Applied only when the aspect is mentioned


Label Convention
----------------
0 : Aspect not mentioned
1 : Mentioned with negative sentiment
2 : Mentioned with positive sentiment
"""


# Model definition

class KcElectraSharedSentiment(nn.Module):
    """
    Shared-encoder model for multi-aspect sentiment analysis.

    Architecture
    ------------
    - One shared KcELECTRA encoder.
    - The [CLS] representation is reused for all aspects.
    - mention_head:
        Linear(hidden_size, num_aspects)
        Trained using multi-label BCE loss.
    - sentiment_heads:
        One Linear(hidden_size, 2) per aspect.
        Trained using cross-entropy loss.
        Applied only when the aspect is mentioned.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier for the pre-trained encoder.
    num_aspects : int, default=4
        Number of aspects to classify.
    dropout : float, default=0.1
        Dropout probability applied to the [CLS] embedding.
    """

    def __init__(self, model_name: str, num_aspects: int = 4, dropout: float = 0.1, mention_pos_weight=None,
        sentiment_class_weights=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout        = nn.Dropout(dropout)
        self.mention_head   = nn.Linear(hidden, num_aspects)
        # One independent linear head per aspect for sentiment (2 classes each)
        self.sentiment_heads = nn.ModuleList(
            [nn.Linear(hidden, 2) for _ in range(num_aspects)]
        )
        
         # register class weights as buffers (non-trainable tensors)
        if mention_pos_weight is not None:
            self.register_buffer("mention_pos_weight", mention_pos_weight)

        if sentiment_class_weights is not None:
            self.register_buffer("sentiment_class_weights", sentiment_class_weights)
            
    def forward(
        self,
        input_ids: torch.Tensor       = None,
        attention_mask: torch.Tensor  = None,
        labels: torch.Tensor          = None,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        input_ids      : (B, L) token ids
        attention_mask : (B, L) attention mask
        labels         : (B, A) integer labels in {0, 1, 2}, optional.
                         Required during training; omit at inference.

        Returns
        -------
        dict with keys:
          "loss"   : scalar training loss (only when labels provided)
          "logits" : (B, 3A) packed logits
                     [:, :A]   â†’ mention logits  (raw, pre-sigmoid)
                     [:, A:]   â†’ sentiment logits reshaped to (B, A*2)
        """
        # Encoder 
        encoder_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls = self.dropout(encoder_out.last_hidden_state[:, 0])  # (B, H)

        # Heads
        mention_logits    = self.mention_head(cls)                          # (B, A)
        sentiment_logits  = torch.stack(
            [head(cls) for head in self.sentiment_heads], dim=1
        )                                                                   # (B, A, 2)

        # Loss (training only)
        loss = None
        if labels is not None:
            loss = _compute_loss(mention_logits, sentiment_logits, labels, self.mention_pos_weight, self.sentiment_class_weights)

        # Pack logits for Trainer compatibility 
        B           = cls.size(0)
        packed      = torch.cat(
            [mention_logits, sentiment_logits.reshape(B, -1)], dim=1
        )                                                                   # (B, 3A)

        return {"loss": loss, "logits": packed} if loss is not None else {"logits": packed}



# Wrapper Class

class SharedABSAWrapper:
    """
    High-level training and inference wrapper for the shared ABSA model.

    This class orchestrates data preparation, training, threshold tuning,
    prediction, and evaluation by delegating the heavy lifting to the
    pure helper functions defined above.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    aspects    : list[str]
        Names of the aspects to classify (become column names in outputs).
    max_length : int
        Maximum token sequence length.
    """
    def __init__(
        self,
        model_name: str,
        aspects: list,
        max_length: int,
        mention_pos_weight=None,
        sentiment_class_weights=None
    ):
        self.aspects    = aspects
        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)

        self.model = KcElectraSharedSentiment(
            model_name=model_name,
            num_aspects=len(self.aspects),
            mention_pos_weight=mention_pos_weight,
            sentiment_class_weights=sentiment_class_weights
        )

        self.trainer    = None

        # Per-aspect mention thresholds; tuned on validation set after training
        self.thresholds = {a: 0.5 for a in self.aspects}
    
    # fit
    def fit(
        self,
        X_train: pd.Series,
        y_train: pd.DataFrame,
        X_val: pd.Series      | None = None,
        y_val: pd.DataFrame   | None = None,
    ) -> "SharedABSAWrapper":
        """
        Train the model and (optionally) tune mention thresholds on validation data.

        Parameters
        ----------
        X_train : training texts
        y_train : training labels DataFrame (columns = aspects, values 0/1/2)
        X_val   : validation texts (optional; skips eval if not provided)
        y_val   : validation label DataFrame (optional)

        Returns
        -------
        self (for method chaining)
        """
        # Build tokenised datasets
        tr_ds = build_dataset(X_train, y_train, self.tokenizer, self.aspects, self.max_length)
        va_ds = (
            build_dataset(X_val, y_val, self.tokenizer, self.aspects, self.max_length)
            if (X_val is not None and y_val is not None)
            else None
        )

        # Training configuration
        args = TrainingArguments(
            output_dir="./kcelectra_shared_absa",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=6,
            eval_strategy="epoch" if va_ds is not None else "no",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="e2e_macro_f1",
            greater_is_better=True,
            fp16=True,
            optim="adamw_torch",      # disable fused optimizer
            logging_strategy="epoch",
            logging_first_step=True,
            report_to="none",
        )

        self.trainer = ContiguousTrainer(
            model           = self.model,
            args            = args,
            train_dataset   = tr_ds,
            eval_dataset    = va_ds,
            compute_metrics = lambda ep: compute_metrics_fn(ep, len(self.aspects)),
        )
        self.trainer.train()
        

        # Tune per-aspect mention thresholds on validation set
        if va_ds is not None:
            packed          = self.trainer.predict(va_ds).predictions
            mention_logits, _ = _unpack_logits(packed, len(self.aspects))
            mention_probs   = _sigmoid(mention_logits)
            self.thresholds = tune_mention_thresholds(
                mention_probs, y_val, self.aspects
            )

        return self

    # --------------------------------------------------------------- predict
    def predict(self, X):
        """
    Run inference on input texts and return aspect sentiment labels.

    Parameters
    ----------
    X : iterable of text
        List or Series of review texts.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (N, len(aspects)) with labels:
        0 = Not Mentioned
        1 = Negative
        2 = Positive
    """    
        self.model.eval()

        enc = self.tokenizer(
            list(X),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # remove token_type_ids if present
        enc.pop("token_type_ids", None)
        
        # detect model device
        device = next(self.model.parameters()).device

        # move tensors to same device
        enc = {k: v.to(device) for k, v in enc.items()}
                
        with torch.no_grad():
            outputs = self.model(**enc)

        packed_logits = outputs["logits"].cpu().numpy()

        return decode_predictions(
            packed_logits,
            self.aspects,
            self.thresholds
        )
        
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # save model weights
        torch.save(self.model.state_dict(), path / "pt")

        # save tokenizer
        self.tokenizer.save_pretrained(path)

        import json

        # save thresholds
        with open(path / "thresholds.json", "w") as f:
            json.dump(self.thresholds, f)

        # save aspects
        with open(path / "aspects.json", "w") as f:
            json.dump(self.aspects, f)