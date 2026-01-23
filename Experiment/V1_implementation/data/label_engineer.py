from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass(frozen=True)
class LabelEngineer:
    config: object

    def compute_progression_slopes(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        if longitudinal_df is None or longitudinal_df.empty:
            return pd.DataFrame()

        min_visits = int(getattr(getattr(self.config, "training", object()), "min_visits_for_slope", 0) or 0)
        totals: List[str] = list(getattr(getattr(self.config, "features", object()), "all_updrs_totals", []) or [])

        if min_visits <= 0 or not totals:
            return pd.DataFrame()

        if "PATNO" not in longitudinal_df.columns or "months_since_baseline" not in longitudinal_df.columns:
            return pd.DataFrame()

        slopes_data = []
        for patno, group in longitudinal_df.groupby("PATNO"):
            valid_group = group.dropna(subset=["months_since_baseline"])
            if len(valid_group) < min_visits:
                continue

            slope_entry: Dict[str, float] = {"PATNO": int(patno), "n_visits": int(len(valid_group))}

            for total in totals:
                if total not in valid_group.columns:
                    continue

                scores = valid_group[total].dropna()
                times_for_total = valid_group.loc[scores.index, "months_since_baseline"].values
                if len(scores) < min_visits:
                    continue

                times_std = pd.Series(times_for_total).std()
                if pd.isna(times_std) or times_std == 0:
                    continue

                result = linregress(times_for_total, scores.values)
                slope_entry[f"{total}_slope"] = float(result.slope)
                slope_entry[f"{total}_r"] = float(result.rvalue)
                slope_entry[f"{total}_p"] = float(result.pvalue)

            if len(slope_entry) > 2:
                slopes_data.append(slope_entry)

        return pd.DataFrame(slopes_data)

    def initialize_slope_dict(self, patnos: List[int]) -> Dict[int, Dict[str, float]]:
        totals: List[str] = list(getattr(getattr(self.config, "features", object()), "all_updrs_totals", []) or [])
        return {int(p): {f"{t}_slope": float("nan") for t in totals} for p in patnos}

    def fill_slope_dict_from_df(self, slopes: Dict[int, Dict[str, float]], slopes_df: pd.DataFrame) -> None:
        if not slopes or slopes_df is None or slopes_df.empty or "PATNO" not in slopes_df.columns:
            return

        totals: List[str] = list(getattr(getattr(self.config, "features", object()), "all_updrs_totals", []) or [])
        valid_patnos = set(slopes.keys())
        slopes_df_filtered = slopes_df[slopes_df["PATNO"].isin(valid_patnos)]

        for _, row in slopes_df_filtered.iterrows():
            patno = int(row["PATNO"])
            if patno not in slopes:
                continue
            for total in totals:
                slope_col = f"{total}_slope"
                if slope_col in slopes_df_filtered.columns and pd.notna(row.get(slope_col)):
                    slopes[patno][slope_col] = float(row[slope_col])

