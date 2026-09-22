"""Example: calculate subject-level, 200-parcel SDI values from fMRI and DWI.

This is a minimal extraction of the SDI calculation in 01_Full_GSP_fMRI.ipynb.
It uses the study's existing utility_GSP.py and utility_functions.py; it does
not reimplement or change their graph-harmonic or SDI calculations.

Required inputs (not included in the public repository):
    * One text file of included participant IDs per group, one ID per line.
    * Subject-level DWI connectomes in the directory structure expected by
      compute_normalized_connectome().
    * Final parcellated resting-state BOLD .npy files, after confound regression,
      temporal filtering, motion censoring and time-point trimming. Each file
      must have shape (time points, 200 parcels), in the same parcel order as SC.

This example starts AFTER the imaging preprocessing and motion-exclusion steps.
If using an earlier BOLD file, reproduce those steps first; otherwise its SDI
values will not reproduce the notebook. BOLD time series are z-scored across
remaining time points here, as in the notebook.

The notebook builds a separate group-average SC matrix for each group; this
example preserves that choice. The lists below should specify the participants
used to construct each group-average SC matrix in the final analysis.

Edit the configuration paths and BOLD filename suffix before running. Keep
participant-level inputs and output files out of the public GitHub repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from utility_functions import compute_normalized_connectome
from utility_GSP import GSP_sub


# ---------- Study-specific configuration: replace with your local paths ----------
N_PARCELS = 200

GROUPS = {
    "control": {
        "subject_list": Path("data/controls.txt"),
        "sc_directory": Path("data/DWI/control"),
    },
    "clinical": {
        "subject_list": Path("data/clinical.txt"),
        "sc_directory": Path("data/DWI/clinical"),
    },
}

BOLD_DIRECTORY = Path("data/parcel_data")
BOLD_SUFFIX = "_task-rest_bold_GS_25.npy"  # Confirm the FINAL file variant.
OUTPUT_DIRECTORY = Path("results/SDI")
# -------------------------------------------------------------------------------


def read_subject_list(path: Path) -> list[str]:
    """Read the exact, ordered list of participants for one group."""
    subjects = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError(f"Empty subject list or repeated subject IDs: {path}")
    return subjects


def load_bold(subject: str) -> np.ndarray:
    """Load final parcel-level BOLD and return z-scored (parcels, time points)."""
    path = BOLD_DIRECTORY / subject / "Schaefer" / f"{subject}{BOLD_SUFFIX}"
    # Input orientation matches the notebook: (time points, parcels).
    bold = np.load(path).T
    if bold.ndim != 2 or bold.shape[0] != N_PARCELS:
        raise ValueError(f"Expected {N_PARCELS} parcels in {path}; got {bold.shape}")
    if not np.isfinite(bold).all():
        raise ValueError(f"Non-finite BOLD values in {path}")
    bold = zscore(bold, axis=1)
    if not np.isfinite(bold).all():
        raise ValueError(f"Invalid z-scores (e.g. constant parcel time series): {path}")
    return bold


def calculate_group_sdi(group: str, subject_list: Path, sc_directory: Path) -> pd.DataFrame:
    """Calculate SDI using the group-average structural connectome for this group."""
    subjects = read_subject_list(subject_list)

    # Same study utility and group-averaging call used by the source notebook.
    group_sc, sc_subject_count = compute_normalized_connectome(
        subjects, str(sc_directory)
    )
    if group_sc.shape != (N_PARCELS, N_PARCELS):
        raise ValueError(f"Unexpected group SC shape for {group}: {group_sc.shape}")
    print(f"{group}: SC connectomes included = {sc_subject_count}")

    results = []
    for subject in subjects:
        bold = load_bold(subject)
        frames = np.arange(bold.shape[1])

        # Retain the original function, output order, and SDI definition.
        split_harmonic, _lff, _hff, sdi, *_ = GSP_sub(group_sc, bold, frames)
        sdi = np.asarray(sdi).reshape(-1)
        if sdi.shape != (N_PARCELS,) or not np.isfinite(sdi).all():
            raise ValueError(f"Unexpected SDI output for {subject}: {sdi.shape}")

        results.append({
            "subject_id": subject,
            "group": group,
            "split_harmonic": split_harmonic,
            **{f"SDI_region_{i + 1}": value for i, value in enumerate(sdi)},
        })

    return pd.DataFrame(results)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tables = [
        calculate_group_sdi(group, paths["subject_list"], paths["sc_directory"])
        for group, paths in GROUPS.items()
    ]
    output = OUTPUT_DIRECTORY / "subject_level_sdi.csv"
    pd.concat(tables, ignore_index=True).to_csv(output, index=False)
    print(f"Saved subject-level SDI values to {output}")


if __name__ == "__main__":
    main()
