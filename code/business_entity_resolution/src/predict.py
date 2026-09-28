import pandas as pd
import xgboost as xgb
import os
import gc

from src import config, preprocess, blocking, features


def run_predictions():

    # ============================================================
    # 1. LOAD TEST DATA
    # ============================================================

    print("--- 1. LOADING TEST SET ---")

    s1 = pd.read_csv(
        f"{config.TEST_DIR}/test_source1.tsv",
        sep="\t"
    )

    # TEST MODE:
    # Only process the first 100,000 Source-1 entities.
    # Source 2 and Source 3 remain fully available as
    # the candidate matching database.

    # Process FULL test Source-1 dataset
    s1 = s1.copy()
    
    print(
        f"Source 1 records being processed: "
        f"{len(s1):,}"
    )

    s2 = pd.read_csv(
        f"{config.TEST_DIR}/test_source2.tsv",
        sep="\t"
    )

    s3 = pd.read_csv(
        f"{config.TEST_DIR}/test_source3.tsv",
        sep="\t"
    )

    # Combine Source 2 and Source 3
    s2s3 = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    # Free original S2/S3 DataFrames
    del s2
    del s3
    gc.collect()


    # ============================================================
    # 2. PREPROCESSING
    # ============================================================

    print("\n--- 2. PREPROCESSING ---")

    s1 = preprocess.preprocess_dataframe(s1)

    s2s3 = preprocess.preprocess_dataframe(s2s3)


    # ============================================================
    # 3. BLOCKING / CANDIDATE GENERATION
    # ============================================================

    print("\n--- 3. BLOCKING (Ultra-Fast) ---")

    # IMPORTANT:
    #
    # blocking.py deletes 'combined_text' from the DataFrames
    # passed into it.
    #
    # Therefore we MUST pass copies here.
    #
    # The original s1 and s2s3 are preserved for feature
    # extraction later.

    s1_blocking = s1[
        ['entity_id', 'combined_text']
    ].copy()

    s2s3_blocking = s2s3[
        ['entity_id', 'combined_text']
    ].copy()

    candidates = blocking.generate_candidate_pairs(
        s1_blocking,
        s2s3_blocking,
        part_num="test"
    )

    # Free blocking copies
    del s1_blocking
    del s2s3_blocking
    gc.collect()

    print(
        f"Generated {len(candidates):,} candidate pairs."
    )


    # ============================================================
    # 4. FEATURE EXTRACTION
    # ============================================================

    print("\n--- 4. FEATURE EXTRACTION ---")

    feat_df = features.generate_features(
        candidates,
        s1,
        s2s3
    )

    # Candidates are no longer required after feature generation
    del candidates
    gc.collect()


    # ============================================================
    # 5. MODEL PREDICTION
    # ============================================================

    print("\n--- 5. PREDICTING ON GPU ---")

    feature_cols = [
        'name_jaro',
        'name_ratio',
        'name_token_sort',
        'name_core_jaro',
        'addr_jaro',
        'addr_ratio',
        'pincode_match',
        'country_match',
        'blocking_score',
        'semantic_similarity'
    ]

    print("Loading XGBoost model...")

    model = xgb.XGBClassifier(
        **config.XGB_PARAMS
    )

    model.load_model(
        f"{config.BASE_DIR}/code/business_entity_resolution/xgb_model.json"
    )

    print("Running predictions...")

    feat_df['match_prob'] = model.predict_proba(
        feat_df[feature_cols]
    )[:, 1]


    # ============================================================
    # 6. FILTER HIGH-CONFIDENCE MATCHES
    # ============================================================

    print(
        f"Applying match threshold: "
        f"{config.MATCH_THRESHOLD}"
    )

    matches = feat_df[
        feat_df['match_prob'] > config.MATCH_THRESHOLD
    ].copy()

    print(
        f"High-confidence candidate matches: "
        f"{len(matches):,}"
    )


    # ============================================================
    # 7. GENERATE OUTPUT
    # ============================================================

    print("\n--- 6. GENERATING SUBMISSION FILE ---")

    # Group candidate matches by Source-1 entity
    match_grouped = (
        matches
        .groupby('source1_entity_id')
        ['candidate_entity_id']
        .apply(lambda x: ','.join(x.astype(str)))
        .reset_index()
    )

    match_grouped.rename(
        columns={
            'candidate_entity_id': 'matched_entity_ids'
        },
        inplace=True
    )


    # Ensure ALL 100,000 Source-1 IDs appear in output
    output_df = pd.DataFrame({
        'source1_entity_id': s1['entity_id']
    })

    output_df = output_df.merge(
        match_grouped,
        on='source1_entity_id',
        how='left'
    )

    output_df['matched_entity_ids'] = (
        output_df['matched_entity_ids']
        .fillna("")
    )


    # ============================================================
    # 8. SAVE RESULT
    # ============================================================

    os.makedirs(
        config.OUTPUT_DIR,
        exist_ok=True
    )

    out_file = os.path.join(
        config.OUTPUT_DIR,
        "matching_results.tsv"
    )

    output_df.to_csv(
        out_file,
        sep="\t",
        index=False
    )

    print(
        f"\nSUCCESS! Final submission file saved to:"
    )

    print(out_file)

    print(
        f"\nOutput rows: "
        f"{len(output_df):,}"
    )


if __name__ == "__main__":
    run_predictions()