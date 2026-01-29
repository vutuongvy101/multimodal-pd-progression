"""
Debug helpers for data inspection.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd


def dump_longitudinal_csv(
    prepared_data: Dict,
    output_path: str,
    max_rows: Optional[int] = None
) -> str:
    """
    Dump the longitudinal DataFrame to CSV for debugging.

    Args:
        prepared_data: Dict with 'longitudinal' DataFrame.
        output_path: CSV output path.
        max_rows: Optional limit on number of rows.

    Returns:
        The resolved output path.
    """
    if 'longitudinal' not in prepared_data or not isinstance(prepared_data['longitudinal'], pd.DataFrame):
        raise ValueError("prepared_data['longitudinal'] must be a pandas DataFrame")

    df = prepared_data['longitudinal']
    if max_rows is not None:
        df = df.head(max_rows)

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)
