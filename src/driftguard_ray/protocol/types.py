"""Shared runtime argument/return type aliases.

本模块仅定义运行时通信签名，保持 tuple 形式以减少对现有业务代码的改动。
"""

from typing import TypeAlias

from driftguard_ray.federate.observation import Observation
from driftguard_ray.federate.params import FedParam
from driftguard_ray.federate.retrain_config import RetrainConfig

# Common aliases
ClientId: TypeAlias = int
TimeStep: TypeAlias = int
Sample: TypeAlias = tuple[bytes, int]
Samples: TypeAlias = list[Sample]

# RPC-compatible tuple signatures
ReqAdvStepArgs: TypeAlias = tuple[ClientId]
ReqAdvStepRes: TypeAlias = tuple[TimeStep]

GetDataArgs: TypeAlias = tuple[ClientId, TimeStep]
GetDataRes: TypeAlias = Samples

ReqUploadObsArgs: TypeAlias = tuple[ClientId, Observation]
ReqUploadObsRes: TypeAlias = tuple[FedParam]

ReqTrigArgs: TypeAlias = tuple[ClientId, Observation, FedParam]
ReqTrigRes: TypeAlias = tuple[FedParam, RetrainConfig]

