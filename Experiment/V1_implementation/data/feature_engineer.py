from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.utils import FeatureScaler


@dataclass
class FeatureEngineer:
    config: object
    normalize_features: bool = True
    static_scaler: FeatureScaler | None = field(default=None, init=False)
    motor_scaler: FeatureScaler | None = field(default=None, init=False)
    updrs_supplementary_scaler: FeatureScaler | None = field(default=None, init=False)
    non_motor_scaler: FeatureScaler | None = field(default=None, init=False)
    medication_scaler: FeatureScaler | None = field(default=None, init=False)
    age_at_visit_scaler: FeatureScaler | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.normalize_features:
            self.static_scaler = FeatureScaler()
            self.motor_scaler = FeatureScaler()
            self.updrs_supplementary_scaler = FeatureScaler()
            self.non_motor_scaler = FeatureScaler()
            self.medication_scaler = FeatureScaler()
            self.age_at_visit_scaler = FeatureScaler()

    def create_missingness_masks(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler: FeatureScaler | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            empty_shape = (len(df), 0)
            empty = np.array([]).reshape(empty_shape).astype(np.float32)
            return empty, empty

        feature_df = df[available_cols].copy()
        for col in available_cols:
            if feature_df[col].dtype == "object":
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            elif feature_df[col].dtype.name == "category":
                feature_df[col] = pd.to_numeric(feature_df[col].cat.codes, errors="coerce")

        values = feature_df.values.astype(np.float32)
        mask = feature_df.isna().values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0)

        if scaler is not None and getattr(scaler, "fitted_", False):
            values = scaler.transform(values, mask, available_cols)

        return values, mask

    def fit_scalers(self, prepared_data: Dict, train_patnos: List[int] | None = None) -> None:
        if not self.normalize_features:
            return

        static_df = prepared_data.get("static", pd.DataFrame())
        longitudinal_df = prepared_data.get("longitudinal", pd.DataFrame())

        if train_patnos is not None:
            static_df = static_df[static_df["PATNO"].isin(train_patnos)]
            longitudinal_df = longitudinal_df[longitudinal_df["PATNO"].isin(train_patnos)]

        static_cols = [c for c in getattr(self.config.features, "static_features", []) if c in static_df.columns]
        if static_cols and self.static_scaler is not None:
            static_values, static_mask = self.create_missingness_masks(static_df, static_cols, scaler=None)
            self.static_scaler.fit_transform(static_values, static_mask, static_cols)

        self._fit_scaler_for_features("motor_features", longitudinal_df, self.motor_scaler)
        self._fit_scaler_for_features("updrs_supplementary_features", longitudinal_df, self.updrs_supplementary_scaler)
        self._fit_scaler_for_features("non_motor_features", longitudinal_df, self.non_motor_scaler)
        self._fit_scaler_for_features("medication_features", longitudinal_df, self.medication_scaler)
        self._fit_scaler_for_features("age_at_visit_features", longitudinal_df, self.age_at_visit_scaler)

    def _fit_scaler_for_features(
        self,
        feature_attr_name: str,
        df: pd.DataFrame,
        scaler: FeatureScaler | None,
    ) -> None:
        if scaler is None or df is None or df.empty:
            return

        features = getattr(self.config.features, feature_attr_name, [])
        if not isinstance(features, (list, tuple, set)) or not features:
            return

        feature_cols = [c for c in features if c in df.columns]
        if not feature_cols:
            return

        values, mask = self.create_missingness_masks(df, feature_cols, scaler=None)
        scaler.fit_transform(values, mask, feature_cols)

    def create_feature_vectors(self, prepared_data: Dict) -> Dict:
        static_df = prepared_data.get("static", pd.DataFrame())
        longitudinal_df = prepared_data.get("longitudinal", pd.DataFrame())

        static_data = self._process_static_features_vectorized(static_df)
        longitudinal_data = self._process_longitudinal_features_vectorized(longitudinal_df)

        return {"static_data": static_data, "longitudinal_data": longitudinal_data}

    def _process_static_features_vectorized(self, static_df: pd.DataFrame) -> Dict[int, Dict]:
        static_cols = [c for c in getattr(self.config.features, "static_features", []) if c in static_df.columns]
        if static_df is None or static_df.empty or "PATNO" not in static_df.columns:
            return {}

        static_df_unique = static_df.drop_duplicates(subset="PATNO", keep="first")
        patient_list = static_df_unique["PATNO"].values
        feature_cols = [c for c in static_cols if c != "PATNO"]

        all_values, all_masks = self.create_missingness_masks(static_df_unique, feature_cols, scaler=self.static_scaler)
        return {
            int(patno): {"values": all_values[i], "mask": all_masks[i]}
            for i, patno in enumerate(patient_list)
        }

    def _get_feature_columns_from_config(self, df: pd.DataFrame, feature_attr_names: List[str]) -> Dict[str, List[str]]:
        result = {}
        for attr_name in feature_attr_names:
            features = getattr(self.config.features, attr_name, [])
            if not isinstance(features, (list, tuple, set)):
                features = []
            result[attr_name] = [c for c in features if c in df.columns]
        return result

    def _extract_all_visit_features(
        self, group: pd.DataFrame, feature_cols: List[str], scaler: FeatureScaler | None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if feature_cols:
            return self.create_missingness_masks(group, feature_cols, scaler=scaler)
        n_visits = len(group)
        empty = np.array([]).reshape(n_visits, 0).astype(np.float32)
        return empty, empty

    def _sort_visits_by_time(self, group: pd.DataFrame) -> pd.DataFrame:
        if "months_since_baseline" in group.columns:
            return group.sort_values("months_since_baseline")
        if "EVENT_ID" in group.columns:
            visit_event_order = {"SC": 0, "BL": 0, **{f"V{i:02d}": i for i in range(1, 26)}}
            default_visit_order = 999
            group = group.copy()
            group["_visit_order"] = group["EVENT_ID"].map(lambda x: visit_event_order.get(x, default_visit_order))
            return group.sort_values("_visit_order").drop(columns=["_visit_order"])
        return group

    def _process_longitudinal_features_vectorized(self, longitudinal_df: pd.DataFrame) -> Dict[int, List[Dict]]:
        if longitudinal_df is None or longitudinal_df.empty or "PATNO" not in longitudinal_df.columns:
            return {}

        feature_cols_dict = self._get_feature_columns_from_config(
            longitudinal_df,
            [
                "motor_features",
                "updrs_supplementary_features",
                "non_motor_features",
                "medication_features",
                "age_at_visit_features",
            ],
        )

        motor_cols = feature_cols_dict["motor_features"]
        updrs_supplementary_cols = feature_cols_dict["updrs_supplementary_features"]
        non_motor_cols = feature_cols_dict["non_motor_features"]
        med_cols = feature_cols_dict["medication_features"]
        age_at_visit_cols = feature_cols_dict.get("age_at_visit_features", [])

        totals = list(getattr(self.config.features, "all_updrs_totals", []) or [])

        longitudinal_data: Dict[int, List[Dict]] = {}
        for patno, group in longitudinal_df.groupby("PATNO"):
            group = self._sort_visits_by_time(group)
            n_visits = len(group)

            all_motor_values, all_motor_masks = self._extract_all_visit_features(group, motor_cols, self.motor_scaler)
            all_updrs_supp_values, all_updrs_supp_masks = self._extract_all_visit_features(
                group, updrs_supplementary_cols, self.updrs_supplementary_scaler
            )
            all_non_motor_values, all_non_motor_masks = self._extract_all_visit_features(
                group, non_motor_cols, self.non_motor_scaler
            )
            all_med_values, all_med_masks = self._extract_all_visit_features(group, med_cols, self.medication_scaler)
            all_age_values, all_age_masks = self._extract_all_visit_features(
                group, age_at_visit_cols, self.age_at_visit_scaler
            )

            time_months_arr = (
                group["months_since_baseline"].fillna(0.0).astype(float).values
                if "months_since_baseline" in group.columns
                else np.zeros(n_visits, dtype=float)
            )

            updrs_totals_arr = np.full((n_visits, len(totals)), np.nan, dtype=float)
            for j, total in enumerate(totals):
                if total in group.columns:
                    updrs_totals_arr[:, j] = pd.to_numeric(group[total], errors="coerce").values

            visits: List[Dict] = []
            for visit_idx in range(n_visits):
                np3tot_val = (
                    float(updrs_totals_arr[visit_idx, 2])
                    if len(totals) > 2 and not np.isnan(updrs_totals_arr[visit_idx, 2])
                    else np.nan
                )
                visits.append(
                    {
                        "motor_values": all_motor_values[visit_idx],
                        "motor_mask": all_motor_masks[visit_idx],
                        "updrs_supplementary_values": all_updrs_supp_values[visit_idx],
                        "updrs_supplementary_mask": all_updrs_supp_masks[visit_idx],
                        "nonmotor_values": all_non_motor_values[visit_idx],
                        "nonmotor_mask": all_non_motor_masks[visit_idx],
                        "med_values": all_med_values[visit_idx],
                        "med_mask": all_med_masks[visit_idx],
                        "age_at_visit_values": all_age_values[visit_idx],
                        "age_at_visit_mask": all_age_masks[visit_idx],
                        "time_months": float(time_months_arr[visit_idx]),
                        "updrs_totals": updrs_totals_arr[visit_idx].copy(),
                        "np3tot": np3tot_val,
                    }
                )

            longitudinal_data[int(patno)] = visits

        return longitudinal_data

