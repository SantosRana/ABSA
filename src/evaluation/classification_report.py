from sklearn.metrics import classification_report, f1_score

def classification_analysis(y_pred, y_true, aspects, labels, label_names):

    per_aspect_f1 = {}

    # -----------------------------
    # Per-aspect evaluation
    # -----------------------------
    for asp in aspects:
        yt = y_true[asp].to_numpy()
        yp = y_pred[asp].to_numpy()

        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        per_aspect_f1[asp] = f1

        
        print(f"\n===== {asp} : Classification Report (3-class) =====")
        print(classification_report(
            yt,
            yp,
            labels=labels,
            target_names=label_names,
            digits=4,
            zero_division=0
        ))

    # -----------------------------
    # Flattened evaluation
    # -----------------------------
    yt_all = y_true.values.ravel()
    yp_all = y_pred.values.ravel()

    macro_f1 = f1_score(yt_all, yp_all, average="macro", zero_division=0)
    micro_f1 = f1_score(yt_all, yp_all, average="micro", zero_division=0)
    
    # Sentiment Only Macro
    mask = yt_all != 0
    macro_no0 = f1_score(yt_all[mask], yp_all[mask], average="macro", zero_division=0)

    # Mention detection evaluation
    yt_m = (yt_all != 0).astype(int)
    yp_m = (yp_all != 0).astype(int)
    mention_macro = f1_score(yt_m, yp_m, average="macro", zero_division=0)
    
    print(f"\nOverall Macro F1: {macro_f1:.4f}")
    print(f"Overall Micro F1: {micro_f1:.4f}")
    print(f"Macro F1 (excluding NOT MENTION): {macro_no0:.4f}")
    print(f"Mention Macro F1: {mention_macro:.4f}")

    print("\n===== OVERALL (all aspects flattened) =====")
    print(classification_report(
        yt_all,
        yp_all,
        labels=labels,
        target_names=label_names,
        digits=4,
        zero_division=0
    ))

    return per_aspect_f1, macro_f1, macro_no0, mention_macro