import gc
import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from sentence_transformers import SentenceTransformer
import torch

from src import config


# ============================================================
# CONFIGURATION
# ============================================================

# Number of candidate pairs processed at one time
CHUNK_SIZE = 20_000


# ============================================================
# MAIN FEATURE GENERATION
# ============================================================

def generate_features(candidate_pairs, df_s1, df_s2s3):

    print("Generating string distance features (Memory Optimized)...")

    # --------------------------------------------------------
    # Create fast dictionary lookups
    # --------------------------------------------------------

    print("Building entity lookup dictionaries...")

    s1_dict = df_s1.set_index("entity_id").to_dict("index")
    s2s3_dict = df_s2s3.set_index("entity_id").to_dict("index")

    total_candidates = len(candidate_pairs)

    print(
        f"Total candidate pairs: {total_candidates:,}"
    )

    # ========================================================
    # STRING / STRUCTURED FEATURES
    # ========================================================

    print("\nCalculating string and structured features...")

    name_jaro = []
    name_ratio = []
    name_token_sort = []
    name_core_jaro = []

    addr_jaro = []
    addr_ratio = []

    pincode_match = []
    country_match = []

    # --------------------------------------------------------
    # Process in chunks so Python doesn't hold unnecessary
    # temporary objects for too long.
    # --------------------------------------------------------

    for start in range(
        0,
        total_candidates,
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            total_candidates
        )

        print(
            f"String features: "
            f"{start:,} to {end:,} "
            f"({end - start:,} pairs)"
        )

        chunk = candidate_pairs.iloc[
            start:end
        ]

        # ----------------------------------------------------
        # itertuples is much faster than iterrows/apply
        # ----------------------------------------------------

        for row in chunk.itertuples(
            index=False
        ):

            s1 = s1_dict[
                str(row.source1_entity_id)
            ]

            s2 = s2s3_dict[
                str(row.candidate_entity_id)
            ]

            # ------------------------------------------------
            # Business names
            # ------------------------------------------------

            n1 = str(
                s1.get(
                    "business_name_clean",
                    ""
                )
            )

            n2 = str(
                s2.get(
                    "business_name_clean",
                    ""
                )
            )

            # ------------------------------------------------
            # Core business names
            # ------------------------------------------------

            n1_core = str(
                s1.get(
                    "business_name_core",
                    ""
                )
            )

            n2_core = str(
                s2.get(
                    "business_name_core",
                    ""
                )
            )

            # ------------------------------------------------
            # Addresses
            # ------------------------------------------------

            a1 = str(
                s1.get(
                    "business_address_clean",
                    ""
                )
            )

            a2 = str(
                s2.get(
                    "business_address_clean",
                    ""
                )
            )

            # ------------------------------------------------
            # Name similarity
            # ------------------------------------------------

            name_jaro.append(
                JaroWinkler.normalized_similarity(
                    n1,
                    n2
                )
            )

            name_ratio.append(
                fuzz.ratio(
                    n1,
                    n2
                ) / 100.0
            )

            name_token_sort.append(
                fuzz.token_sort_ratio(
                    n1,
                    n2
                ) / 100.0
            )

            name_core_jaro.append(
                JaroWinkler.normalized_similarity(
                    n1_core,
                    n2_core
                )
            )

            # ------------------------------------------------
            # Address similarity
            # ------------------------------------------------

            addr_jaro.append(
                JaroWinkler.normalized_similarity(
                    a1,
                    a2
                )
            )

            addr_ratio.append(
                fuzz.ratio(
                    a1,
                    a2
                ) / 100.0
            )

            # ------------------------------------------------
            # Pincode
            # ------------------------------------------------

            p1 = s1.get(
                "pincode",
                ""
            )

            p2 = s2.get(
                "pincode",
                ""
            )

            if (
                pd.notna(p1)
                and pd.notna(p2)
                and str(p1) != ""
                and str(p2) != ""
            ):

                if str(p1) == str(p2):
                    pincode_match.append(1)
                else:
                    pincode_match.append(-1)

            else:
                pincode_match.append(0)

            # ------------------------------------------------
            # Country
            # ------------------------------------------------

            c1 = s1.get(
                "country",
                ""
            )

            c2 = s2.get(
                "country",
                ""
            )

            if pd.isna(c1):
                c1 = ""

            if pd.isna(c2):
                c2 = ""

            country_match.append(
                int(
                    str(c1) == str(c2)
                )
            )

        del chunk

        gc.collect()

    # ========================================================
    # CREATE STRING FEATURE DATAFRAME
    # ========================================================

    features_df = pd.DataFrame({
        "name_jaro": name_jaro,
        "name_ratio": name_ratio,
        "name_token_sort": name_token_sort,
        "name_core_jaro": name_core_jaro,
        "addr_jaro": addr_jaro,
        "addr_ratio": addr_ratio,
        "pincode_match": pincode_match,
        "country_match": country_match
    })

    # Free lists
    del name_jaro
    del name_ratio
    del name_token_sort
    del name_core_jaro
    del addr_jaro
    del addr_ratio
    del pincode_match
    del country_match

    gc.collect()

    # ========================================================
    # SEMANTIC FEATURES
    # ========================================================

    print("\nGenerating semantic embeddings (GPU Accelerated)...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Load MiniLM
    # --------------------------------------------------------

    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        device=device
    )

    semantic_scores = []

    # ========================================================
    # CHUNKED SEMANTIC PROCESSING
    # ========================================================

    print(
        "\nProcessing semantic similarity "
        "in memory-safe chunks..."
    )

    for start in range(
        0,
        total_candidates,
        CHUNK_SIZE
    ):

        end = min(
            start + CHUNK_SIZE,
            total_candidates
        )

        print(
            f"Semantic features: "
            f"{start:,} to {end:,} "
            f"({end - start:,} pairs)"
        )

        chunk = candidate_pairs.iloc[
            start:end
        ]

        # ----------------------------------------------------
        # Collect texts
        # ----------------------------------------------------

        s1_texts = []
        s2_texts = []

        for row in chunk.itertuples(
            index=False
        ):

            s1 = s1_dict[
                str(row.source1_entity_id)
            ]

            s2 = s2s3_dict[
                str(row.candidate_entity_id)
            ]

            text1 = s1.get(
                "combined_text",
                ""
            )

            text2 = s2.get(
                "combined_text",
                ""
            )

            if pd.isna(text1):
                text1 = ""

            if pd.isna(text2):
                text2 = ""

            s1_texts.append(
                str(text1)
            )

            s2_texts.append(
                str(text2)
            )

        # ----------------------------------------------------
        # Encode Source 1
        # ----------------------------------------------------

        emb1 = model.encode(
            s1_texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # ----------------------------------------------------
        # Encode candidate Source 2/3
        # ----------------------------------------------------

        emb2 = model.encode(
            s2_texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # ----------------------------------------------------
        # Cosine similarity
        #
        # Embeddings are normalized, therefore:
        #
        # cosine similarity = dot product
        # ----------------------------------------------------

        similarity = np.sum(
            emb1 * emb2,
            axis=1
        )

        semantic_scores.extend(
            similarity.astype(
                np.float32
            ).tolist()
        )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        del s1_texts
        del s2_texts
        del emb1
        del emb2
        del similarity
        del chunk

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # ADD SEMANTIC FEATURE
    # ========================================================

    features_df["semantic_similarity"] = (
        semantic_scores
    )

    del semantic_scores
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # COMBINE WITH CANDIDATE INFORMATION
    # ========================================================

    print("\nCombining all features...")

    result = pd.concat(
        [
            candidate_pairs.reset_index(
                drop=True
            ),
            features_df.reset_index(
                drop=True
            )
        ],
        axis=1
    )

    # ========================================================
    # VERIFY EXACT TRAINING FEATURES
    # ========================================================

    expected_features = [
        "name_jaro",
        "name_ratio",
        "name_token_sort",
        "name_core_jaro",
        "addr_jaro",
        "addr_ratio",
        "pincode_match",
        "country_match",
        "blocking_score",
        "semantic_similarity"
    ]

    missing = [
        col
        for col in expected_features
        if col not in result.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required model features: {missing}"
        )

    print(
        "\nFeature generation complete!"
    )

    print(
        f"Candidate pairs: "
        f"{len(result):,}"
    )

    print(
        "\nModel features:"
    )

    for col in expected_features:
        print(
            f"  ✓ {col}"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    del features_df
    del s1_dict
    del s2s3_dict

    gc.collect()

    return result