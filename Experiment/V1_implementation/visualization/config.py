from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class VizConfig:
    target: Optional[str] = "NP3TOT"
    dt_buckets: Tuple[str, ...] = ("dt_0_6", "dt_6_12", "dt_12_24", "dt_24_inf")
    dpi: int = 150
