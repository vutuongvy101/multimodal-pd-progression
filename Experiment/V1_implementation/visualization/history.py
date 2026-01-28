import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

@dataclass
class History:
    path: str
    data: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "History":
        # with open(path, "r") as f:
        #     d = json.load(f)
        # return History(path=path, data=d)
        resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"training_history.json not found.\n"
                f"  given: {path}\n"
                f"  resolved: {resolved}\n"
                f"  cwd: {os.getcwd()}"
            )
        with open(resolved, "r") as f:
            d = json.load(f)
        return History(path=resolved, data=d)

    def series(self, key: str) -> Optional[np.ndarray]:
        v = self.data.get(key)
        if v is None:
            return None
        return np.asarray(v, dtype=float)

    def n_epochs(self) -> int:
        for k in ("val_loss_next_visit", "val_loss", "train_loss"):
            s = self.series(k)
            if s is not None:
                return int(len(s))
        return 0

    def best_epoch(self) -> int:
        """0-indexed. Prefer val_loss_next_visit if present."""
        arr = self.series("val_loss_next_visit")
        if arr is None:
            arr = self.series("val_loss")
        if arr is None or len(arr) == 0:
            return 0
        return int(np.argmin(arr))

    def val_metrics_epoch(self, epoch_idx: int) -> Optional[Dict[str, Any]]:
        vm = self.data.get("val_metrics")
        if vm is None:
            return None
        if isinstance(vm, list):
            return vm[epoch_idx] if 0 <= epoch_idx < len(vm) else None
        if isinstance(vm, dict):
            return vm.get(str(epoch_idx)) or vm.get(epoch_idx)
        return None

    def pick_target(self, preferred: Optional[str]) -> Optional[str]:
        if preferred:
            return preferred
        e = self.best_epoch()
        vme = self.val_metrics_epoch(e)
        if not vme:
            return None
        nv = vme.get("next_visit", {})
        if isinstance(nv, dict) and len(nv) > 0:
            return sorted(nv.keys())[0]
        return None
