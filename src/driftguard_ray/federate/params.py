from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, List, TypeAlias
import numpy as np
import torch
import torch.nn as nn
from driftguard_ray.config import get_logger
from driftguard_ray.model.utils import freeze_layer, unfreeze_layer

logger = get_logger("params")

Params: TypeAlias = List[np.ndarray]

class ParamType(Enum):
    """Parameter names used for retraining aggregation."""
    DG_FULL = "dg_full"
    DG_PARTIAL = "dg_partial"
    DG_TOGETHER = "dg_together"
    FULL = "full"
    CLUSTER = "cluster"
    MOE = "moe"
    NONE = None

    _GATE = "gate"
    

@dataclass
class FedParam:
    LOCAL_SIZE: ClassVar[int] = 0
    GATE_SIZE: ClassVar[int] = 0
    OTHER_SIZE: ClassVar[int] = 0
    # 实例变量
    gate: Params = field(default_factory=list)
    local: Params = field(default_factory=list)
    other: Params = field(default_factory=list)
    # dg_shared: Params = field(default_factory=list)
    # dg_gate: Params = field(default_factory=list)
    # local: Params = field(default_factory=list)
    # full: Params = field(default_factory=list)
    # moe_shared: Params = field(default_factory=list)
    def is_empty(self) -> bool:
        return not (self.gate or self.local or self.other)
    @staticmethod
    def freeze_exclude(model: nn.Module, param_type: ParamType) -> None:
        # Driftguare
        if param_type == ParamType.DG_FULL:
            # keep oters, gate, freeze local
            freeze_layer(model, include_names=["local"])
            pass
        elif param_type == ParamType.DG_PARTIAL:
            # keep local, gate freeze others
            freeze_layer(model, include_names=["local", "gate"], exclude= True)
        # FedAvg, PFL, Cluster
        elif param_type == ParamType.FULL or param_type == ParamType.CLUSTER or param_type == ParamType.MOE:
            pass
        elif param_type == ParamType._GATE:
            freeze_layer(model, include_names=["gate"], exclude= True)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")
    @staticmethod
    def unfreeze(model: nn.Module, include_names: List[str] = []) -> None:
        if include_names:
            unfreeze_layer(model, include_names=include_names)
        else:
            unfreeze_layer(model)

    def set(self, model: nn.Module)-> None:
        """Set model parameters from FedParam instance."""
        if self.gate:
            set_params(model, self.gate, names=["gate"])
        if self.local:
            set_params(model, self.local, names=["local"])
        if self.other:
            set_params(model, self.other, names=["local", "gate"], exclude=True)
    # @staticmethod
    # def set(model: nn.Module, params: Params, param_type: ParamType) -> None:
    #     if param_type == ParamType.DG_FULL:
    #         set_params(model, params, names=["local"], exclude=True)
    #         # len_local = FedParam.LOCAL_SIZE or len(
    #         #     FedParam.get(model, ParamType.DG_PARTIAL)
    #         # )
    #         # local, shared = params[:len_local], params[len_local:]
    #         # set_params(model, local, names=["local"])
    #         # if shared:
    #             # set_params(model, shared, names=["local", "gate"], exclude=True)
    #     elif param_type == ParamType.MOE:
    #         set_params(model, params, names=["local"], exclude=True)
    #     elif param_type == ParamType.DG_PARTIAL:
    #         gate_params, local_params = (
    #             params[: FedParam.GATE_SIZE],
    #             params[FedParam.GATE_SIZE :],
    #         )
    #         set_params(model, gate_params, names=["gate"])
    #         set_params(model, local_params, names=["local"])
    #     elif param_type == ParamType.FULL or param_type == ParamType.CLUSTER:
    #         set_params(model, params)
    #     else:
    #         raise ValueError(f"Unknown param_type: {param_type}")
    @staticmethod
    def get(model: nn.Module) -> FedParam:
        gate_params = get_params(model, names=["gate"])
        local_params = get_params(model, names=["local"])
        other_params = get_params(model, names=["local", "gate"], exclude=True)
        if not FedParam.GATE_SIZE:
            FedParam.GATE_SIZE = len(gate_params)
        if not FedParam.LOCAL_SIZE:
            FedParam.LOCAL_SIZE = len(local_params)
        if not FedParam.OTHER_SIZE:
            FedParam.OTHER_SIZE = len(other_params)
        return FedParam(gate=gate_params, local=local_params, other=other_params)
    # @staticmethod
    # def get(model: nn.Module, param_type: ParamType, trig: bool = True) -> Params:
    #     if param_type == ParamType.DG_FULL: 
    #         return get_params(model, names=["local"], exclude=True)
    #         # return get_params(model, names=["local", "gate"], exclude=True)
    #         # local = get_params(model, names=["local"])
    #         # shared = get_params(model, names=["local", "gate"], exclude=True)
    #         # return [*local, *shared]
    #     elif param_type == ParamType.DG_PARTIAL:
    #         # return [gate, local]
    #         if FedParam.GATE_SIZE == 0:
    #             FedParam.GATE_SIZE = len(get_params(model, names=["gate"]) )
    #         gate_params = get_params(model, names=["gate"])
    #         local_params = get_params(model, names=["local"]) if trig else []
    #         return [
    #             *gate_params,
    #             *local_params,
    #         ]
    #     elif param_type == ParamType.MOE:
    #         return get_params(model, names=["local"], exclude=True)
    #         # return get_params(model, names=["local", "gate"], exclude=True)
    #     elif param_type == ParamType.FULL or param_type == ParamType.CLUSTER:
    #         return get_params(model)
    #     else:
    #         raise ValueError(f"Unknown param_type: {param_type}")
    def merge(self) -> Params:
        """Merge all params into a single list."""
        assert self.gate and self.local and self.other, "FedParam merge: missing params"
        return [*self.gate, *self.local, *self.other]
    @staticmethod
    def separate(params: Params) -> FedParam:
        """Separate a single list of params into a FedParam instance."""
        assert FedParam.GATE_SIZE and FedParam.LOCAL_SIZE and FedParam.OTHER_SIZE, (
            "FedParam separate: size not set"
        )
        assert len(params) == (
            FedParam.GATE_SIZE + FedParam.LOCAL_SIZE + FedParam.OTHER_SIZE
        ), "FedParam separate: mismatched length"
        gate = params[: FedParam.GATE_SIZE]
        local = params[FedParam.GATE_SIZE : FedParam.GATE_SIZE + FedParam.LOCAL_SIZE]
        other = params[FedParam.GATE_SIZE + FedParam.LOCAL_SIZE :]
        return FedParam(gate=gate, local=local, other=other)

def aggregate_params(params_list: List[Params], sample_sizes: List[int] = []) -> Params:
    """Aggregate model parameters (FedAvg)."""
    if not params_list:
        raise ValueError("No parameters to aggregate")

    if len(params_list) == 1:
        return params_list[0]

    # Calculate total samples
    total_samples = sum(sample_sizes)
    if total_samples == 0:
        # If no sample info, use average aggregation
        total_samples = len(params_list)
        sample_sizes = [1] * len(params_list)

    # Initialize aggregated parameters
    aggregated_params = []

    # Aggregate each layer
    for layer_idx in range(len(params_list[0])):
        # Weighted average
        layer_sum = None

        for client_params, sample_size in zip(params_list, sample_sizes):
            weight = sample_size / total_samples
            layer_params = client_params[layer_idx]

            if layer_sum is None:
                layer_sum = layer_params * weight
            else:
                layer_sum += layer_params * weight

        aggregated_params.append(layer_sum)

    return aggregated_params


def get_params(model: nn.Module, names: List[str] = [], exclude: bool = False) -> Params:
    """Get model weights as a list of numpy arrays.

    Args:
        model: The model to extract weights from.
    """
    params = []
    for layer_name, param in model.named_parameters():
        if exclude:
            if names and all(name not in layer_name for name in names):
                params.append(param.data.cpu().numpy())
        else:
            if any(name in layer_name for name in names) or not names:
                params.append(param.data.cpu().numpy()) 
    return params

def set_params(
    model: nn.Module, params: Params, names: List[str] = [], exclude: bool = False
) -> None:
    """Set model weights from a list of numpy arrays.
    """

    # Get parameter names and their corresponding parameters
    # logger.debug(f"layer_name: {layer_name}, params len: {len(params)}")
    if exclude:
        param_names = [
            layer_name
            for layer_name, _ in model.named_parameters()
            if names and all(name not in layer_name for name in names)
        ]
    else:
        param_names = [
            layer_name
            for layer_name, _ in model.named_parameters()
            if any(name in layer_name for name in names) or not names
        ]
    assert len(param_names) == len(params), "set_params: Mismatched number"

    # Create state_dict from numpy weights
    state_dict = {
        n: torch.from_numpy(p) for n, p in zip(param_names, params)
    }
    model.load_state_dict(state_dict, strict=False)
