import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def error_analysis(y_pred, y_true, x_text, aspects, top_n_errors=20):
    x_text = pd.Series(x_text).reset_index(drop=True)
    errs = []
    for i in range(len(x_text)):
        for asp in aspects:
            t = int(y_true.iloc[i][asp])
            p = int(y_pred.iloc[i][asp])
            if t != p:
                errs.append({
                    "row_id": i,
                    "text": x_text.loc[i],
                    "aspect": asp,
                    "true_label": t,
                    "pred_label": p
                })

    err_df = pd.DataFrame(errs)
    print(f"\nTotal misclassified aspect-decisions: {len(err_df)}")

    if len(err_df):
        print(err_df.head(top_n_errors))
        print("\nErrors by aspect:")
        print(err_df["aspect"].value_counts())

    return err_df