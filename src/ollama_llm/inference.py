import re 
import ast
import pandas as pd
from tqdm import tqdm
from .build_prompt import build_prompt
from .query import query_qwen

def parse_prediction(text):
    """
    Parse model output string to extract prediction vector.

    Args:
        text (str): Raw text response from the model.

    Returns:
        list[int]: A list of four integers representing predictions
                   for aspects [FOOD, PRICE, SERVICE, AMBIENCE].
    """
    # Look for a pattern like [0-2,0-2,0-2,0-2] in the text
    match = re.search(r"\[[0-2],[0-2],[0-2],[0-2]\]", text)

    if match:
        # Safely evaluate the matched string into a Python list
        return ast.literal_eval(match.group())

    # Fallback: return neutral predictions if no match found
    return [0, 0, 0, 0]

def predict_llm_batch(X):
    """
    Generate predictions for a batch of input texts using Qwen.

    Args:
        X (str | list | pd.Series): Input review(s)
    Returns:
        pd.DataFrame: DataFrame with predictions for each aspect.
    """
    # Convert input to list
    if isinstance(X, str):
        texts = [X]
    elif isinstance(X, pd.Series):
        texts = X.tolist()
    else:
        texts = list(X)

    predictions = []

    for text in tqdm(texts, disable=len(texts) == 1):
        # Build prompt for the model
        prompt = build_prompt(text)

        # Query the model
        raw = query_qwen(prompt)

        # Parse raw response into prediction vector
        pred = parse_prediction(raw)

        # Collect predictions
        predictions.append(pred)

    # Return predictions as a structured DataFrame
    return pd.DataFrame(
        predictions,
        columns=["FOOD", "PRICE", "SERVICE", "AMBIENCE"]
    )

