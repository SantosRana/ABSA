import joblib
import torch
import json
from transformers import AutoTokenizer
from pathlib import Path

# Root Path
ROOT = Path(__file__).resolve().parents[2]

def load_models():
    """
    Loads and initializes two models:
    1. A pre-trained Logistic Regression model (saved with joblib).
    2. A fine-tuned KcELECTRA model wrapped for Aspect-Based Sentiment Analysis (ABSA).

    Returns:
        tuple: (lr_model, kc_model)
            - lr_model: Logistic Regression model
            - kc_model: KcELECTRA ABSA model
    """

    # Load the Logistic Regression model from a pickle file
    lr_model = joblib.load(ROOT / "weights" / "lr_model.pkl")

    # Import the wrapper for the KcELECTRA model
    from kc_electra.model import SharedABSAWrapper

    # Initialize the KcELECTRA model with specific aspects and configuration
    kc_model = SharedABSAWrapper(
        model_name="beomi/KcELECTRA-base-v2022",   # Pre-trained base model
        aspects=["FOOD", "PRICE", "SERVICE", "AMBIENCE"],  # Aspects for sentiment analysis
        max_length=128,  # Maximum token length for input sequences
    )

    # Load the saved model weights into memory
    state_dict = torch.load(ROOT / "weights" / "kc_electra" / "kc_electra.pt", map_location="cpu")

    # Remove unused keys to avoid errors when loading state_dict
    state_dict.pop("mention_pos_weight", None)
    state_dict.pop("sentiment_class_weights", None)

    # Load the weights into the model (non-strict to allow missing keys)
    kc_model.model.load_state_dict(state_dict, strict=False)

    # Move the model to CPU for inference
    kc_model.model.to("cpu")

    # Load tokenizer from the saved directory
    kc_model.tokenizer = AutoTokenizer.from_pretrained(ROOT / "weights" / "kc_electra")

    # Load threshold values for classification decisions
    with open(ROOT / "weights" / "kc_electra" / "thresholds.json", "r", encoding="utf-8") as f:
        kc_model.thresholds = json.load(f)

    # Set the model to evaluation mode (disables dropout, etc.)
    kc_model.model.eval()

    # Return both models for use
    return lr_model, kc_model