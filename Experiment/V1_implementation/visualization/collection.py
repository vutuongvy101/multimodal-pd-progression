import glob
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .history import History

@dataclass
class HistoryCollection:
    histories: List[History]

    @staticmethod
    def from_glob(pattern: str) -> "HistoryCollection":
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No files matched: {pattern}")
        return HistoryCollection([History.load(p) for p in paths])

    def stack_series(self, key: str) -> Optional[np.ndarray]:
        """(n_folds, n_epochs_min) truncated to min length across folds."""
        series_list = [h.series(key) for h in self.histories]
        series_list = [s for s in series_list if s is not None]
        if not series_list:
            return None
        min_len = min(len(s) for s in series_list)
        return np.stack([s[:min_len] for s in series_list], axis=0)
