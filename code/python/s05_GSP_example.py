"""Example usage: motion scrubbing and subject-level SDI from parcellated fMRI.

This is a short, standalone *usage example* extracted from 01_Full_GSP_fMRI.ipynb.
The study's existing utility files perform structural-connectome averaging and
GSP/SDI calculations; their implementations are NOT repeated here.

Edit the paths and BOLD_SUFFIX for your local, non-public imaging dataset.
The input BOLD arrays must have shape (time points, 200 parcels), with the same
parcel order as the structural connectomes. The fMRIPrep confounds TSV must
correspond to the same run and original time points.

IMPORTANT: The motion-processing order below follows the uploaded notebook:
identify high-FD frames in the middle of the scan, delete those original frame
indices, THEN trim the first/last 24 of the remaining BOLD frames. That is not
equivalent to trimming BOLD and FD together before scrubbing. Verify this order,
the motion-exclusion cutoff and the chosen BOLD file against the exact pipeline
that generated the final manuscript results before presenting the example as a
reproduction of the published analysis.

Participant data and generated outputs should remain outside the public repo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from utility_functions import compute_normalized_connectome
from utility_GSP import GSP_sub


# ---- Local inputs / analysis settings (confirm against the final analysis) ----
N_PARCELS = 200
TRIM_START = 24
TRIM_END = 24
FD_THRESHOLD_MM = 0.5
MAX_MOTION_FRACTION = 0.15   # Source notebook: >15% of 720 original frames.
ORIGINAL_FRAMES = 720

SUBJECT_LISTS = {
    "control": Path("data/controls.txt"),
    "clinical": Path("data/clinical.txt"),
}
SC_DIRECTORIES = {
    "control": Path("data/DWI/control"),
    "clinical": Path("data/DWI/clinical"),
}
BOLD_DIRECTORY = Path("data/parcel_data")
BOLD_SUFFIX = "_task-rest_bold_gs.npy"  # Confirm the exact final input file.
CONFOUNDS_DIRECTORY = Path("data/fmriprep")
CONFOUNDS_SUFFIX = "_task-rest_desc-confounds_timeseries.tsv"
OUTPUT_CSV = Path("results/subject_level_sdi.csv")
# ---------------------------------------------------------------------------


sdi_rows = []

for group, subject_list_path in SUBJECT_LISTS.items():
    subjects = [s.strip() for s in subject_list_path.read_text().splitlines() if s.strip()]
    if not subjects:
        raise ValueError(f"No subjects found in {subject_list_path}")

    # Same group-average SC utility and call as in the source notebook.
    group_sc, sc_count = compute_normalized_connectome(
        subjects, str(SC_DIRECTORIES[group])
    )
    if np.asarray(group_sc).shape != (N_PARCELS, N_PARCELS):
        raise ValueError(f"Unexpected group SC dimensions for {group}")
    print(f"{group}: structural connectomes used in the average = {sc_count}")

    for subject in subjects:
        bold_path = BOLD_DIRECTORY / subject / "Schaefer" / f"{subject}{BOLD_SUFFIX}"
        fd_path = (
            CONFOUNDS_DIRECTORY / subject / "func"
            / f"{subject}{CONFOUNDS_SUFFIX}"
        )

        # fMRIPrep framewise displacement, indexed by original scan volume.
        fd = pd.read_csv(fd_path, sep="\t")["framewise_displacement"].fillna(0)
        bold = np.load(bold_path)  # original (time points, parcels)
        if bold.ndim != 2 or bold.shape != (len(fd), N_PARCELS):
            raise ValueError(
                f"{subject}: BOLD {bold.shape} does not match confounds "
                f"({len(fd)} frames) and {N_PARCELS} parcels. "
                "Check whether this BOLD file was previously trimmed/scrubbed."
            )

        fd_middle = fd.iloc[TRIM_START:-TRIM_END]
        high_motion_indices = fd_middle[fd_middle > FD_THRESHOLD_MM].index.to_numpy()
        n_high_motion = len(high_motion_indices)
        if n_high_motion > ORIGINAL_FRAMES * MAX_MOTION_FRACTION:
            print(f"Skipping {subject}: {n_high_motion} high-motion frames")
            continue

        # Preserve the SOURCE NOTEBOOK'S order of censoring and edge trimming.
        bold = np.delete(bold.T, high_motion_indices, axis=1)  # (parcels, frames)
        bold = bold[:, TRIM_START:-TRIM_END]
        bold = zscore(bold, axis=1)
        if bold.shape[1] < 2 or not np.isfinite(bold).all():
            raise ValueError(f"{subject}: invalid BOLD time series after scrubbing")

        frames = np.arange(bold.shape[1])
        split_harmonic, _lff, _hff, sdi, *_ = GSP_sub(group_sc, bold, frames)
        sdi = np.asarray(sdi).reshape(-1)
        if sdi.shape != (N_PARCELS,) or not np.isfinite(sdi).all():
            raise ValueError(f"{subject}: unexpected SDI output: {sdi.shape}")

        sdi_rows.append({
            "subject_id": subject,
            "group": group,
            "split_harmonic": split_harmonic,
            **{f"SDI_region_{i + 1}": float(value) for i, value in enumerate(sdi)},
        })

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(sdi_rows).to_csv(OUTPUT_CSV, index=False)
print(f"Saved SDI for {len(sdi_rows)} participants to {OUTPUT_CSV}")
