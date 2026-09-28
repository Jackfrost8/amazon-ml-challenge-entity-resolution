import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

FEATURE_COLS = [
    "name_jaro",
    "name_ratio",
    "name_token_sort",
    "name_core_jaro",
    "addr_jaro",
    "addr_ratio",
    "pincode_match",
    "country_match",
    "blocking_score",
    "semantic_similarity",
]

DEFAULT_THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
]


# ============================================================
# F0.5
# ============================================================

def f05_score(precision, recall):
    """
    F0.5:
        (1 + beta^2) * P * R / (beta^2 * P + R)

    beta = 0.5
    beta^2 = 0.25
    """
    denominator = (0.25 * precision) + recall

    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path):
    print("\nLoading ground truth...")

    gt = pd.read_csv(path, sep="\t")

    required_cols = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    for col in required_cols:
        if col not in gt.columns:
            raise ValueError(
                f"Ground truth is missing required column: {col}"
            )

    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    # Convert:
    #
    # S2-123,S3-456
    #
    # into:
    #
    # {"S2-123", "S3-456"}
    def parse_matches(value):
        if not isinstance(value, str) or not value.strip():
            return set()

        return {
            x.strip()
            for x in value.split(",")
            if x.strip()
        }

    gt["true_matches"] = gt["matched_entity_ids"].apply(parse_matches)

    return gt


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(path):
    print("\nLoading feature file...")
    print(f"Path: {path}")

    df = pd.read_csv(path)

    print(f"Feature rows: {len(df):,}")

    required_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ] + FEATURE_COLS

    missing = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    return df


# ============================================================
# LOAD XGBOOST
# ============================================================

def load_model(model_path):
    print("\nLoading XGBoost model...")
    print(f"Path: {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    return model


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

def generate_probabilities(df, model):
    print("\nGenerating XGBoost probabilities...")

    X = df[FEATURE_COLS]

    probabilities = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["probability"] = probabilities

    print(
        f"Probability range: "
        f"{probabilities.min():.6f} - "
        f"{probabilities.max():.6f}"
    )

    return df


# ============================================================
# CALCULATE MACRO F0.5
# ============================================================

def evaluate_threshold(df, ground_truth, threshold):

    # --------------------------------------------------------
    # Predictions above threshold
    # --------------------------------------------------------

    predicted = df[df["probability"] >= threshold]

    # Build:
    #
    # source1_entity_id -> set(candidate_entity_id)
    #
    prediction_dict = {}

    for s1_id, group in predicted.groupby("source1_entity_id"):
        prediction_dict[s1_id] = set(
            group["candidate_entity_id"]
        )

    # --------------------------------------------------------
    # Evaluate EVERY Source-1 entity
    #
    # This is important because the challenge uses a
    # macro-average across Source-1 entities and singletons
    # matter.
    # --------------------------------------------------------

    scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    entities_with_prediction = 0

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]

        true_matches = row["true_matches"]

        predicted_matches = prediction_dict.get(
            s1_id,
            set()
        )

        if predicted_matches:
            entities_with_prediction += 1

        # True positives
        tp = len(
            predicted_matches.intersection(true_matches)
        )

        # False positives
        fp = len(
            predicted_matches - true_matches
        )

        # False negatives
        fn = len(
            true_matches - predicted_matches
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # ----------------------------------------------------
        # Per-entity precision
        # ----------------------------------------------------

        if len(predicted_matches) == 0:

            if len(true_matches) == 0:
                # Correct singleton prediction
                precision = 1.0
                recall = 1.0

            else:
                # Missed all true matches
                precision = 0.0
                recall = 0.0

        else:

            precision = (
                tp / len(predicted_matches)
            )

            if len(true_matches) == 0:
                recall = 0.0
            else:
                recall = (
                    tp / len(true_matches)
                )

        # ----------------------------------------------------
        # Per-entity F0.5
        # ----------------------------------------------------

        score = f05_score(
            precision,
            recall
        )

        scores.append(score)

    # --------------------------------------------------------
    # Macro F0.5
    # --------------------------------------------------------

    macro_f05 = (
        np.mean(scores)
        if scores
        else 0.0
    )

    # --------------------------------------------------------
    # Global precision / recall
    #
    # These are useful for diagnostics, but macro F0.5 above
    # is the competition-style metric.
    # --------------------------------------------------------

    if total_tp + total_fp > 0:
        global_precision = (
            total_tp /
            (total_tp + total_fp)
        )
    else:
        global_precision = 0.0

    if total_tp + total_fn > 0:
        global_recall = (
            total_tp /
            (total_tp + total_fn)
        )
    else:
        global_recall = 0.0

    predicted_pairs = len(predicted)

    return {
        "threshold": threshold,
        "f05": macro_f05,
        "precision": global_precision,
        "recall": global_recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_pairs": predicted_pairs,
        "entities_with_prediction": entities_with_prediction,
        "total_entities": len(ground_truth),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Validate Entity Resolution model using macro F0.5"
    )

    parser.add_argument(
        "--features",
        default="features_part1.csv",
        help="Path to features CSV"
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to train_ground_truth.tsv"
    )

    parser.add_argument(
        "--model",
        default="xgb_model.json",
        help="Path to XGBoost model"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            ".."
        )
    )

    project_dir = os.path.join(
        base_dir,
        "code",
        "business_entity_resolution"
    )

    train_dir = os.path.join(
        base_dir,
        "dataset",
        "train"
    )

    # If paths are not absolute, resolve relative to project dir
    if not os.path.isabs(args.features):
        features_path = os.path.join(
            project_dir,
            args.features
        )
    else:
        features_path = args.features

    if args.ground_truth is None:
        ground_truth_path = os.path.join(
            train_dir,
            "train_ground_truth.tsv"
        )
    elif not os.path.isabs(args.ground_truth):
        ground_truth_path = os.path.join(
            project_dir,
            args.ground_truth
        )
    else:
        ground_truth_path = args.ground_truth

    if not os.path.isabs(args.model):
        model_path = os.path.join(
            project_dir,
            args.model
        )
    else:
        model_path = args.model

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENTITY RESOLUTION MODEL VALIDATION")
    print("=" * 70)

    print("\nFiles:")
    print(f"Features     : {features_path}")
    print(f"Ground truth : {ground_truth_path}")
    print(f"Model        : {model_path}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        features_path,
        ground_truth_path,
        model_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nFile not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    df = load_features(features_path)

    gt = load_ground_truth(
        ground_truth_path
    )

    model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # Generate probabilities ONCE
    #
    # Then we can test many thresholds without running
    # XGBoost again.
    # --------------------------------------------------------

    df = generate_probabilities(
        df,
        model
    )

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD EXPERIMENT")
    print("=" * 70)

    results = []

    for threshold in DEFAULT_THRESHOLDS:

        print(
            f"\nEvaluating threshold "
            f"{threshold:.2f}..."
        )

        result = evaluate_threshold(
            df,
            gt,
            threshold
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("VALIDATION RESULTS")
    print("=" * 90)

    print(
        f"{'Threshold':>10} "
        f"{'F0.5':>12} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'TP':>10} "
        f"{'FP':>10} "
        f"{'FN':>10} "
        f"{'Predicted':>12}"
    )

    print("-" * 90)

    for _, row in results_df.iterrows():

        print(
            f"{row['threshold']:>10.2f} "
            f"{row['f05']:>12.6f} "
            f"{row['precision']:>12.6f} "
            f"{row['recall']:>12.6f} "
            f"{int(row['tp']):>10,} "
            f"{int(row['fp']):>10,} "
            f"{int(row['fn']):>10,} "
            f"{int(row['predicted_pairs']):>12,}"
        )

    # --------------------------------------------------------
    # Best threshold according to F0.5
    # --------------------------------------------------------

    best = results_df.loc[
        results_df["f05"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST THRESHOLD BY MACRO F0.5")
    print("=" * 70)

    print(
        f"\nThreshold : {best['threshold']:.2f}"
    )

    print(
        f"F0.5      : {best['f05']:.6f}"
    )

    print(
        f"Precision : {best['precision']:.6f}"
    )

    print(
        f"Recall    : {best['recall']:.6f}"
    )

    print(
        f"TP        : {int(best['tp']):,}"
    )

    print(
        f"FP        : {int(best['fp']):,}"
    )

    print(
        f"FN        : {int(best['fn']):,}"
    )

    print(
        f"Predicted : {int(best['predicted_pairs']):,}"
    )

    print(
        f"Entities with prediction: "
        f"{int(best['entities_with_prediction']):,} / "
        f"{int(best['total_entities']):,}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = os.path.join(
        project_dir,
        "validation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nValidation results saved to:\n"
        f"{output_path}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()