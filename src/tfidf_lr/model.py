import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
import time


class ABSATwoStageLR:
    """
    2-stage ABSA baseline:
    Stage 1: aspect mention detection (0 vs non-0), multi-head LR
    Stage 2: per-aspect sentiment classification (1 vs 2), LR on mentioned rows
    """

    def __init__(
        self,
        aspects=("FOOD", "PRICE", "SERVICE", "AMBIENCE"),
        tfidf_params=None,
        lr_params=None,
        threshold_grid=None,
    ):
        self.aspects = list(aspects)

        self.tfidf_params = tfidf_params or dict(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=50000,
            min_df=2,
        )
        self.lr_params = lr_params or dict(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )
        self.threshold_grid = (
            np.arange(0.10, 0.91, 0.05) if threshold_grid is None else np.array(threshold_grid)
        )

        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.mention_model = None
        self.sent_models = {}     # aspect -> {"model": clf or None, "fallback_class": int or None}
        self.thresholds = {}      # aspect -> float
        self.is_fitted = False

    def _to_mentions(self, y_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert aspect sentiment labels into binary mention indicators.

        Parameters
        ----------
        y_df : pd.DataFrame
            DataFrame containing sentiment labels for each aspect.
            Each column corresponds to an aspect, with values:
              - 0 meaning "not mentioned"
              - non-zero meaning "mentioned" with some sentiment

        Returns
        -------
        pd.DataFrame
            Binary DataFrame of the same shape, where:
              - 1 indicates the aspect is mentioned
              - 0 indicates the aspect is not mentioned
        """
        # For each aspect column, check if value != 0 (mentioned)
        # Convert boolean result to integers (True → 1, False → 0)
        return (y_df[self.aspects] != 0).astype(int)

    def fit(self, X_train, y_train: pd.DataFrame, X_val=None, y_val=None):
        
        # Convert training text data into numerical feature vectors
        Xtr = self.vectorizer.fit_transform(X_train)

        # -------------------------------
        # Stage 1: Mention detection model
        # -------------------------------
        # Convert training labels into binary "mention" indicators (aspect mentioned or not)
        ytr_m = self._to_mentions(y_train)

        # Train a multi-output classifier (one logistic regression per aspect)
        self.mention_model = MultiOutputClassifier(LogisticRegression(**self.lr_params))
        self.mention_model.fit(Xtr, ytr_m)

        # -------------------------------
        # Threshold tuning (optional)
        # -------------------------------
        # If validation data is provided, tune thresholds for each aspect
        if X_val is not None and y_val is not None:
            Xva = self.vectorizer.transform(X_val)
            yva_m = self._to_mentions(y_val)

            # For each aspect, find the threshold that maximizes F1 score
            for j, asp in enumerate(self.aspects):
                # Predicted probabilities for the positive class
                p = self.mention_model.estimators_[j].predict_proba(Xva)[:, 1]
                y_true = yva_m[asp].values

                best_f1, best_t = -1.0, 0.5
                for t in self.threshold_grid:
                    y_pred = (p >= t).astype(int)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, float(t)
                # Store best threshold for this aspect
                self.thresholds[asp] = best_t
        else:
            # Default threshold = 0.5 if no validation set provided
            self.thresholds = {asp: 0.5 for asp in self.aspects}

        # -------------------------------
        # Stage 2: Sentiment classification
        # -------------------------------
        # Train sentiment classifiers only on samples where the aspect is mentioned
        self.sent_models = {}
        for asp in self.aspects:
            # Select rows where aspect is mentioned (non-zero label)
            m = y_train[asp] != 0
            Xtr_a = Xtr[m.values]
            ytr_a = y_train.loc[m, asp].values

            # If only one sentiment class exists, store fallback class
            uniq = np.unique(ytr_a)
            if len(uniq) < 2:
                self.sent_models[asp] = {"model": None, "fallback_class": int(uniq[0])}
            else:
                # Otherwise, train a logistic regression sentiment classifier
                clf = LogisticRegression(**self.lr_params)
                clf.fit(Xtr_a, ytr_a)
                self.sent_models[asp] = {"model": clf, "fallback_class": None}

        # Mark model as fitted
        self.is_fitted = True
        return self
    
    def predict(self, X):

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = pd.Series(X)  # ensure pandas series

        Xs = self.vectorizer.transform(X.tolist())

        n = Xs.shape[0]

        out = np.zeros((n, len(self.aspects)), dtype=int)

        for j, asp in enumerate(self.aspects):

            p_m = self.mention_model.estimators_[j].predict_proba(Xs)[:, 1]

            y_m = (p_m >= self.thresholds[asp]).astype(int)

            if self.sent_models[asp]["model"] is None:
                y_s = np.full(n, self.sent_models[asp]["fallback_class"], dtype=int)

            else:
                y_s = self.sent_models[asp]["model"].predict(Xs)

            out[:, j] = np.where(y_m == 1, y_s, 0)

        return pd.DataFrame(out, columns=self.aspects, index=X.index)