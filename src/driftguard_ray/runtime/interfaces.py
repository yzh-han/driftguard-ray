"""Runtime endpoint interfaces.

这些接口用于解耦客户端训练逻辑与底层通信实现（Ray、未来其他实现）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from driftguard_ray.protocol.types import (
    GetDataArgs,
    GetDataRes,
    ReqAdvStepArgs,
    ReqAdvStepRes,
    ReqTrigArgs,
    ReqTrigRes,
    ReqUploadObsArgs,
    ReqUploadObsRes,
)


class DataServiceEndpoint(Protocol):
    """Data service endpoint contract."""

    def get_data(self, args: GetDataArgs) -> GetDataRes:
        """Fetch client data samples.

        Args:
            args: Tuple (client_id, time_step).
        Returns:
            Sample list.
        """
        ...

    def stop(self) -> bool:
        """Request service stop.

        Returns:
            Whether stop signal is accepted.
        """
        ...


class ServerEndpoint(Protocol):
    """Federated server endpoint contract."""

    def req_adv_step(self, args: ReqAdvStepArgs) -> ReqAdvStepRes:
        """Request advancing global step."""
        ...

    def req_upload_obs(self, args: ReqUploadObsArgs) -> ReqUploadObsRes:
        """Upload client observation and receive params."""
        ...
    def req_trig(self, args: ReqTrigArgs) -> ReqTrigRes:
        """Request retrain trigger result."""
        ...

    def stop(self) -> bool:
        """Request server stop."""
        ...


@dataclass
class RuntimeHandle:
    """Runtime handle that exposes endpoints and close hook."""

    data_endpoint: DataServiceEndpoint
    server_endpoint: ServerEndpoint

