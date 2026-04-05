import pandas as pd

def build_labels(group):
    """
    Convert aspect-based sentiment annotations into multi-class labels.

    Args:
        group (pd.DataFrame): Group of rows corresponding to one sentence.

    Returns:
        pd.Series: A dictionary-like object with aspect labels:
                   0 = aspect not mentioned
                   1 = aspect mentioned with negative sentiment
                   2 = aspect mentioned with positive sentiment
    """
    
    # Define the aspects of interest for restaurant reviews
    aspects = ["FOOD", "PRICE", "SERVICE", "AMBIENCE"]
    
    # Initialize all aspects as "not mentioned" (0)
    label_dict = {aspect: 0 for aspect in aspects}
    
    # Iterate through rows (triplets) for the sentence
    for _, row in group.iterrows():
        sentiment = row["sentiment"]
        
        # Assign sentiment-based labels
        if sentiment == "negative":
            label_dict[row["aspect"]] = 1
        elif sentiment == "positive":
            label_dict[row["aspect"]] = 2
    
    # Return as a Series
    return pd.Series(label_dict)