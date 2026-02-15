
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Tuple, TypeVar

from driftguard_ray.federate.observation import Observation
from driftguard_ray.federate.params import FedParam, ParamType, Params
from driftguard_ray.federate.retrain_config import RetrainConfig


T = TypeVar("T")


@dataclass
class Req(Generic[T]):
    """Acknowledge result with a typed message payload."""

    recv: bool = False
    payload: T | None = None

class ReqState:
    """Track per-client acknowledgments for server time steps."""

    def __init__(self, num_clients: int) -> None:
        """Initialize per-client time step acknowledgments."""
        self.step: dict[int, Req[Any]] = {
            c: Req() for c in range(num_clients)
        } # cid -> Req[Any]
        self.obs: dict[int, Req[Observation]] = {
            c: Req() for c in range(num_clients)
        }
        # 
        self.trig: dict[int, Req[Tuple[Observation, FedParam]]] = {
            c: Req() for c in range(num_clients)
        }
    
    def all_recv(self, type: str) -> bool:
        """Check if all clients have received the acknowledgment of a given type.

        Args:
            type: Type of acknowledgment ("step", "obs", "trig").

        Returns:
            True if all clients have received, False otherwise.
        """
        if type == "step":
            return all(req.recv for req in self.step.values())
        elif type == "obs":
            return all(req.recv for req in self.obs.values())
        elif type == "trig":
            return all(req.recv for req in self.trig.values())
        else:
            raise ValueError(f"ReqState: Unknown acknowledgment type: {type}")
    
    def reset(self) -> None:
        """Reset all acknowledgment states for the next round."""
        for req in self.step.values():
            req.recv = False
            req.payload = None
        for req in self.obs.values():
            req.recv = False
            req.payload = None
        for req in self.trig.values():
            req.recv = False
            req.payload = None


@dataclass
class RetrainState:
    class Stage(Enum):
            IDLE = 0
            ONGOING = 1
            COMPLETED = 2

    _rt_round: int  # config
    def __post_init__(self):
        self.remain_round: int = 0  # remain
        self.rt_cfg: RetrainConfig = RetrainConfig(False, [], ParamType.FULL)
        self.is_cluster: bool = False
    @property
    def stage(self) -> Stage:
        if self.remain_round > 0:
            return self.Stage.ONGOING
        else:
            if self.rt_cfg.trigger:
                return self.Stage.COMPLETED
            else:
                # idle: remain 0 and not trigger
                return self.Stage.IDLE
            
        
            
