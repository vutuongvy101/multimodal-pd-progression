from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def get_df_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Lightweight DataFrame summary helper for tests."""
    n_rows = len(df)
    return {
        "n_rows": n_rows,
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "missing_pct": (df.isnull().sum() / n_rows * 100) if n_rows else (df.isnull().sum() * 0),
    }

