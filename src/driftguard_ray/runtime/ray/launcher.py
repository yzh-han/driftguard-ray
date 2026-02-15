"""Ray runtime launcher."""

from dataclasses import dataclass

import ray

from driftguard_ray.data.service import DataServiceArgs
from driftguard_ray.federate.server.fed_server import FedServerArgs
from driftguard_ray.runtime.interfaces import RuntimeHandle
from driftguard_ray.runtime.ray.actors import (
    RayDataServiceActor,
    RayDataServiceEndpoint,
    RayFedServerActor,
    RayServerEndpoint,
)


@dataclass
class RayRuntimeConfig:
    """Ray runtime bootstrap config."""

    data_args: DataServiceArgs
    server_args: FedServerArgs
    ray_address: str | None = None
    log_to_driver: bool = True


def start_ray_runtime(cfg: RayRuntimeConfig) -> RuntimeHandle:
    """Start ray runtime and return endpoint handle.

    Args:
        cfg: Runtime bootstrap config.
    Returns:
        RuntimeHandle with data/server endpoints and close callback.
    """
    owns_ray = False
    

    data_actor = RayDataServiceActor.remote(cfg.data_args)
    data_ep = RayDataServiceEndpoint(data_actor)

    # Inject data endpoint to server args for stop chaining.
    cfg.server_args.data_endpoint = data_ep
    server_actor = RayFedServerActor.options(max_concurrency=32).remote(cfg.server_args)
    server_ep = RayServerEndpoint(server_actor)
    # print("Ray runtime started: fed server...s")

    def close() -> None:
        """Stop actors and shutdown ray if owned by this launcher."""
        try:
            server_ep.stop()
        finally:
            try:
                data_ep.stop()
            finally:
                if owns_ray and ray.is_initialized():
                    ray.shutdown()

    return RuntimeHandle(
        data_endpoint=data_ep,
        server_endpoint=server_ep,
    )
