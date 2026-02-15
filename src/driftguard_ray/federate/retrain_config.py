
from dataclasses import dataclass
from typing import List

from driftguard_ray.federate.params import ParamType

@dataclass
class RetrainConfig:
    trigger: bool
    selection: List[int] # selected clients
    param_type: ParamType
