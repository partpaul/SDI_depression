"""Example: covariate-adjusted, parcel-wise group comparison for any regional measure.

Edit the CONFIGURATION block and run: python s06_group_differences_general.py

Uses the same group-label permutation procedure as the supplied analysis script.
Your final published analysis should document the exact input data and permutation
scheme used; this example does not implement network/global aggregation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from tqdm import trange


# ============================== CONFIGURATION ===============================
DATA_FILE = Path("path/to/main_glm_df.csv")  # One row per participant
OUTPUT_DIR = Path("results/group_comparisons")
OUTPUT_NAME = "SDI_pure_MDD_vs_HC"

# Change the prefix to that of your FC/SC columns, or supply explicit column names.
FEATURE_PREFIX = "SDI_region_"
FEATURE_COLUMNS = None  # e.g., ["SDI_region_1", "SDI_region_2", ...]

# Positive t-values mean GROUP_1 > GROUP_0, after covariate adjustment.
GROUP_COLUMN = "group_y"
GROUP_0 = "control"
GROUP_1 = "clinical"

# Optional filtering BEFORE the two groups are compared.
# Example for medication/comorbidity comparisons: {"group_y": "clinical"}
COHORT_FILTERS = {}

# Optional: restrict only GROUP_1 to IDs from a subgroup CSV (e.g., pure MDD).
# The subgroup file only needs a column called SUBJECT_ID_COLUMN.
GROUP1_SUBJECTS_FILE = None  # Path("path/to/pure_mdd_sample_n78.csv")
SUBJECT_ID_COLUMN = "sub_id"

# For group comparisons of gender itself, remove "gender" from this tuple.
COVARIATES = ("age", "gender", "Mean_FD")
N_PERMUTATIONS = 1000
RANDOM_SEED = 42
FDR_ALPHA = 0.05
# =============================================================================


def run_comparison():
    df = pd.read_csv(DATA_FILE, dtype={SUBJECT_ID_COLUMN: "string"})
    required = {SUBJECT_ID_COLUMN, GROUP_COLUMN, *COVARIATES, *COHORT_FILTERS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Columns absent from DATA_FILE: {missing}")

    for column, value in COHORT_FILTERS.items():
        df = df.loc[df[column] == value].copy()

    df = df.loc[df[GROUP_COLUMN].isin([GROUP_0, GROUP_1])].copy()
    if GROUP1_SUBJECTS_FILE is not None:
        subset = pd.read_csv(
            GROUP1_SUBJECTS_FILE, usecols=[SUBJECT_ID_COLUMN],
            dtype={SUBJECT_ID_COLUMN: "string"},
        )
        selected_ids = set(subset[SUBJECT_ID_COLUMN].dropna())
        df = df.loc[
            (df[GROUP_COLUMN] == GROUP_0)
            | ((df[GROUP_COLUMN] == GROUP_1)
               & df[SUBJECT_ID_COLUMN].isin(selected_ids))
        ].copy()

    if df[SUBJECT_ID_COLUMN].isna().any() or df[SUBJECT_ID_COLUMN].duplicated().any():
        raise ValueError("Participant IDs must be present and unique after filtering.")

    feature_cols = (
        list(FEATURE_COLUMNS) if FEATURE_COLUMNS is not None
        else [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    )
    if not feature_cols:
        raise ValueError(f"No features found for prefix {FEATURE_PREFIX!r}.")
    absent = sorted(set(feature_cols).difference(df.columns))
    if absent:
        raise ValueError(f"Feature columns absent from DATA_FILE: {absent}")

    # Explicitly exclude participants with missing covariates, group, or features.
    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in COVARIATES:
        if column != "gender":
            df[column] = pd.to_numeric(df[column], errors="raise")
    before = len(df)
    df = df.dropna(subset=[GROUP_COLUMN, *COVARIATES, *feature_cols]).copy()
    print(f"Excluded for missing covariates/features: {before - len(df)}")

    counts = df[GROUP_COLUMN].value_counts()
    n0, n1 = int(counts.get(GROUP_0, 0)), int(counts.get(GROUP_1, 0))
    if not n0 or not n1:
        raise ValueError(f"Both groups must be present (got {GROUP_0}: {n0}, {GROUP_1}: {n1}).")
    print(f"{OUTPUT_NAME}: {GROUP_1} (n={n1}) vs {GROUP_0} (n={n0})")

    X = pd.DataFrame(index=df.index)
    X["intercept"] = 1.0
    X["group"] = (df[GROUP_COLUMN] == GROUP_1).astype(float)
    for column in COVARIATES:
        if column == "gender":
            observed = set(df[column].unique())
            if not observed.issubset({"Male", "Female"}):
                raise ValueError(f"Unexpected gender labels: {observed}")
            X[column] = (df[column] == "Male").astype(float)
        else:
            X[column] = df[column].astype(float)

    design = X.to_numpy(dtype=float)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Rank-deficient design: remove redundant covariates or check groups.")
    Y = df[feature_cols].to_numpy(dtype=float)  # participants × regional features
    if len(df) <= design.shape[1]:
        raise ValueError("Not enough participants for the covariate-adjusted GLM.")

    group_idx = X.columns.get_loc("group")
    observed_t = np.array([
        sm.OLS(Y[:, i], design).fit().tvalues[group_idx]
        for i in range(Y.shape[1])
    ])

    rng = np.random.default_rng(RANDOM_SEED)
    null_t = np.empty((N_PERMUTATIONS, Y.shape[1]), dtype=float)
    original_group = design[:, group_idx].copy()
    for p in trange(N_PERMUTATIONS, desc="Permuting group labels"):
        perm_design = design.copy()
        perm_design[:, group_idx] = rng.permutation(original_group)
        xtx_inv = np.linalg.pinv(perm_design.T @ perm_design)
        coefficients = (xtx_inv @ perm_design.T) @ Y
        residuals = Y - perm_design @ coefficients
        mse = np.sum(residuals ** 2, axis=0) / (len(df) - design.shape[1])
        se = np.sqrt(mse * xtx_inv[group_idx, group_idx])
        null_t[p] = coefficients[group_idx] / se

    empirical_p = (np.sum(np.abs(null_t) >= np.abs(observed_t), axis=0) + 1) / (
        N_PERMUTATIONS + 1
    )
    significant, fdr_p = fdrcorrection(empirical_p, alpha=FDR_ALPHA)
    filtered_t = np.where(significant, observed_t, 0.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "feature": feature_cols,
        "t_group1_minus_group0": observed_t,
        "p_permutation_two_sided": empirical_p,
        "p_fdr": fdr_p,
        "significant_fdr": significant,
    })
    results.to_csv(OUTPUT_DIR / f"{OUTPUT_NAME}_results.csv", index=False)
    np.save(OUTPUT_DIR / f"{OUTPUT_NAME}_t_map.npy", observed_t)
    np.save(OUTPUT_DIR / f"{OUTPUT_NAME}_t_map_FDR_filtered.npy", filtered_t)
    print(f"FDR-significant features: {significant.sum()} / {len(feature_cols)}")


if __name__ == "__main__":
    run_comparison()

