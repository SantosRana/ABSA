import pandas as pd
import json
import re

def is_korean(text):
    """
    Check if a given text contains Korean characters.

    Args:
        text (str): Input string.

    Returns:
        bool: True if the text contains at least one Korean character,
              False otherwise.
    """
    return bool(re.search("[\uac00-\ud7a3]", text))


def safe_parse_triplets(triplets_str):
    """
    Safely parse a string representing aspect-based sentiment triplets.

    The dataset stores triplets as strings that look like lists.
    This function attempts to parse them into Python lists using JSON.

    Args:
        triplets_str (str): Raw string containing triplets.

    Returns:
        list: Parsed triplets if successful, otherwise an empty list.
    """
    try:
        # Replace single quotes with double quotes for valid JSON parsing
        return json.loads(triplets_str.replace("'", '"'))
    except:
        # Return empty list if parsing fails
        return []


def process_split(split, dataset):
    """
    Process a dataset split (train/validation/test) to extract Korean samples.


    Args:
        split (str): Name of the dataset split ("train", "validation", "test").
        dataset (dict-like): Dataset object containing splits with text examples.


    Returns:
        pd.DataFrame: DataFrame containing structured rows with
                      sentence, aspect_term, category, and sentiment.
    """
    rows = []  # Collect processed rows
    
    for example in dataset[split]:
        raw = example["text"]
        
        # Skip if no triplet delimiter is present
        # The delimiter "####" separates the sentence from its triplets.
        if "####" not in raw:
            continue
            
        # Split into sentence and triplet string
        sentence, triplets_str = raw.split("####", 1)
        
        # Skip if triplets are empty
        if triplets_str.strip() == "[]":
            continue
        
        # Keep only Korean sentences
        if not is_korean(sentence):
            continue
        
        # Parse triplets safely
        triplets = safe_parse_triplets(triplets_str)
        
        # Each triplet should have 3 elements: aspect, category, sentiment
        for triplet in triplets:
            if len(triplet) == 3:
                rows.append({
                    "sentence": sentence.strip(),
                    "aspect_term": triplet[0],
                    "category": triplet[1],
                    "sentiment": triplet[2]
                })
    
    return pd.DataFrame(rows)