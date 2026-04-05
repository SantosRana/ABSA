from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def split_data(X, y, aspects):

    # Step 1: Split into Train (70%) and Temp (30%)
    # -------------------------
    # MultilabelStratifiedShuffleSplit ensures that label distribution
    # is preserved across splits for multilabel classification problems.
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,        # Perform only one split
        test_size=0.3,     # 30% of data goes into temporary set
        random_state=42   
    )

    # Perform the first split
    for train_idx, temp_idx in msss1.split(X, y):
        X_train = X.iloc[train_idx]   # Training features (70%)
        y_train = y.iloc[train_idx]   # Training labels (70%)
        X_temp = X.iloc[temp_idx]     # Temporary features (30%)
        y_temp = y.iloc[temp_idx]     # Temporary labels (30%)

    
    # Step 2: Split Temp into Validation (15%) and Test (15%)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,        # Perform only one split
        test_size=0.5,     # 50% of temp data → test, 50% → validation
        random_state=42    # Fix random seed for reproducibility
    )

    # Perform the second split
    for val_idx, test_idx in msss2.split(X_temp, y_temp):
        X_val = X_temp.iloc[val_idx]   # Validation features (15%)
        y_val = y_temp.iloc[val_idx]   # Validation labels (15%)
        X_test = X_temp.iloc[test_idx] # Test features (15%)
        y_test = y_temp.iloc[test_idx] # Test labels (15%)

    # Return the three sets: Train (70%), Validation (15%), Test (15%)
    return X_train, y_train, X_val, y_val, X_test, y_test