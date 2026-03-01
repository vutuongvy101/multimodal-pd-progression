"""
Select PATNOs from longitudinal_debug.csv based on visit counts and UPDRS data coverage.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

UPDRS_TOTAL_COLS = ["NP1RTOT", "NP2PTOT", "NP3TOT", "NP4TOT"]
COHORT_COLS = [
    "COHORT",
    "ENROLL_CAT",
    "APPRDX",
    "DIAGNOSIS",
    "SEX",
    "GENDER",
    "RACE",
    "ETHNICITY",
    "AGE_AT_VISIT",
]


def _available_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _select_patnos_from_df(
    df: pd.DataFrame,
    output_path: str,
    min_visits: int = 3,
    include_cohort_cols: bool = True,
) -> str:
    if "PATNO" not in df.columns:
        raise ValueError("Data must include PATNO column")

    # Visit counts per PATNO
    visit_counts = df.groupby("PATNO").size()
    eligible_min = visit_counts[visit_counts >= min_visits]

    min_patno = int(eligible_min.idxmin()) if not eligible_min.empty else None
    max_patno = int(visit_counts.idxmax()) if not visit_counts.empty else None

    # Least data per UPDRS total
    updrs_least: Dict[str, Optional[int]] = {}
    for col in UPDRS_TOTAL_COLS:
        if col not in df.columns:
            updrs_least[col] = None
            continue
        counts = df[df[col].notna()].groupby("PATNO").size()
        updrs_least[col] = int(counts.idxmin()) if not counts.empty else None

    # Total UPDRS score per visit and patient mean
    available_updrs_cols = _available_cols(df, UPDRS_TOTAL_COLS)
    if available_updrs_cols:
        df["UPDRS_TOTAL_SUM"] = df[available_updrs_cols].sum(axis=1, skipna=True)
        patient_mean = df.groupby("PATNO")["UPDRS_TOTAL_SUM"].mean()
        lowest_total_patno = int(patient_mean.idxmin()) if not patient_mean.empty else None
        highest_total_patno = int(patient_mean.idxmax()) if not patient_mean.empty else None
    else:
        lowest_total_patno = None
        highest_total_patno = None

    # Cohort description columns (if present)
    cohort_cols = _available_cols(df, COHORT_COLS) if include_cohort_cols else []

    def describe_patno(patno: Optional[int]) -> Dict[str, Optional[str]]:
        if patno is None:
            return {c: None for c in cohort_cols}
        row = df[df["PATNO"] == patno]
        if row.empty:
            return {c: None for c in cohort_cols}
        # Use first non-null value for each cohort column
        return {c: row[c].dropna().iloc[0] if row[c].notna().any() else None for c in cohort_cols}

    rows = []

    def add_row(category: str, patno: Optional[int], extra: Optional[str] = None):
        desc = describe_patno(patno)
        rows.append({
            "category": category,
            "patno": patno,
            "extra": extra,
            **desc,
        })

    add_row(f"min_visits_ge_{min_visits}", min_patno)
    add_row("max_visits", max_patno)

    for col, patno in updrs_least.items():
        add_row(f"least_data_{col}", patno)

    add_row("lowest_updrs_total_mean", lowest_total_patno)
    add_row("highest_updrs_total_mean", highest_total_patno)

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return str(out_path)


def select_patnos_from_longitudinal_csv(
    csv_path: str,
    output_path: str,
    min_visits: int = 3
) -> str:
    """
    Compute PATNO selections from a longitudinal CSV and write a summary CSV.

    Selections:
      - min visits (>=min_visits)
      - max visits
      - least data for UPDRS I/II/III/IV
      - lowest total UPDRS score (sum of available totals)
      - highest total UPDRS score (sum of available totals)

    Args:
        csv_path: Path to longitudinal_debug.csv
        output_path: Path to write summary CSV
        min_visits: Minimum visits required for "min visits" selection

    Returns:
        Resolved output path.
    """
    df = pd.read_csv(csv_path)
    return _select_patnos_from_df(
        df=df,
        output_path=output_path,
        min_visits=min_visits,
        include_cohort_cols=True,
    )


def select_patnos_from_predictions(
    actuals: np.ndarray,
    patno: np.ndarray,
    output_path: str,
    min_visits: int = 3
) -> str:
    """
    Compute PATNO selections from validation predictions/targets (in-memory).

    Args:
        actuals: Ground truth values [n_visits, n_targets]
        patno: PATNO per visit [n_visits]
        output_path: Path to write summary CSV
        min_visits: Minimum visits required for "min visits" selection

    Returns:
        Resolved output path.
    """
    actuals_np = np.asarray(actuals)
    patno_np = np.asarray(patno)

    if actuals_np.ndim != 2:
        raise ValueError("actuals must be 2D [n_visits, n_targets]")
    if patno_np.ndim != 1:
        raise ValueError("patno must be 1D [n_visits]")
    if actuals_np.shape[0] != patno_np.shape[0]:
        raise ValueError("actuals and patno must have the same number of visits")

    target_cols = UPDRS_TOTAL_COLS[:actuals_np.shape[1]]
    df = pd.DataFrame(actuals_np, columns=target_cols)
    df.insert(0, "PATNO", patno_np)

    return _select_patnos_from_df(
        df=df,
        output_path=output_path,
        min_visits=min_visits,
        include_cohort_cols=False,
    )


def export_validation_dataset_csv(
    validation_df: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Export validation dataset as longitudinal CSV.

    Args:
        validation_df: DataFrame containing validation dataset with patient data
        output_path: Path to write validation dataset CSV

    Returns:
        Resolved output path.
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(out_path, index=False)
    return str(out_path)


def export_validation_dataset_from_csv(
    validation_csv_path: str,
    output_filename: str = "validation_dataset_longitudinal.csv",
) -> str:
    """
    Export validation dataset from CSV to longitudinal CSV format.
    Saves output to the same directory as the input CSV.

    Args:
        validation_csv_path: Path to validation dataset CSV
        output_filename: Filename for the output CSV (saved in same directory as input)

    Returns:
        Resolved output path.
    """
    df = pd.read_csv(validation_csv_path)
    input_dir = Path(validation_csv_path).parent
    output_path = input_dir / output_filename
    return export_validation_dataset_csv(
        validation_df=df,
        output_path=str(output_path),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Select PATNOs from longitudinal_debug.csv")
    p.add_argument("csv_path", help="Path to longitudinal_debug.csv")
    p.add_argument("--output", default="patno_selection_summary.csv", help="Output CSV path")
    p.add_argument("--min-visits", type=int, default=3, help="Minimum visits for min selection")
    args = p.parse_args()

    out = select_patnos_from_longitudinal_csv(
        csv_path=args.csv_path,
        output_path=args.output,
        min_visits=args.min_visits
    )
    print(f"Saved PATNO summary to {out}")
