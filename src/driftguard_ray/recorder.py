
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import os
import json

@dataclass
class Recorder:
    name: str
    root: str = "exp"

    def __post_init__(self) -> None:
        self.metrics: Dict[str, List] = {
            "acc": [],
            "cost": [],
        } 
    
    def update_acc(self, time_step: int, acc: float) -> None:
        self.metrics["acc"].append((time_step, acc))
    
    def update_cost(self, time_step: int, trainable_params: float, epochs: int) -> None:
        self.metrics["cost"].append((time_step, trainable_params, epochs))

    def record(self, cid:int) -> None:
        dir_path = Path(self.root) / self.name
        os.makedirs(dir_path, exist_ok=True)
        path = Path(dir_path) / f"c_{cid}.json"
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)