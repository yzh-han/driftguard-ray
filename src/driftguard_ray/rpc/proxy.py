from driftguard_ray.federate.observation import Observation
from driftguard_ray.federate.params import FedParam, ParamType, Params
from driftguard_ray.federate.retrain_config import RetrainConfig
from driftguard_ray.rpc.rpc import Node, RPCClient
from typing import Any, Callable, Optional, Tuple, List, Dict



class DataServiceProxy:
    """
    Data service proxy for federated learning clients.
    """

    def __init__(self, node: Node):
        """Initialize the data service proxy."""
        # Initialize the RPC client with the data service configuration
        self.dataservice_rpc = RPCClient(node)

    def get_data(self, args: Tuple[int, int,]) -> List[Tuple[bytes, int]]:
        """Get a batch of data samples for a client at a given time step.
        Args:
            args: Tuple containing (client_id, time_step).
        Returns:
            List of samples: List[Tuple[bytes, int]]
        """

        @self.dataservice_rpc.call
        def get_data(args):
            pass

        return get_data(args)

    def stop(self) -> None:

        @self.dataservice_rpc.call
        def stop() -> None:
            pass

        return stop()


class ServerProxy:
    """Server proxy for federated learning clients."""

    def __init__(self, node: Node):
        self.server_rpc = RPCClient(node)

    def req_adv_step(self, args: Tuple[int]) -> Tuple[int,]:
        """
        (client_id,) -> (time_step,)
        """
        @self.server_rpc.call
        def req_adv_step(args):
            pass

        return req_adv_step(args)
    
    def req_upload_obs(self, args: Tuple[int, Observation]) -> Tuple[FedParam,]:
        """
        (client_id, obs) -> (params,)
        """
        @self.server_rpc.call
        def req_upload_obs(args):
            pass

        return req_upload_obs(args)
    
    def req_trig(self, args: Tuple[int, Observation, FedParam]) -> Tuple[FedParam, RetrainConfig]:
        """
        (client_id, obs) -> None
        """
        @self.server_rpc.call
        def req_trig(args):
            pass

        return req_trig(args)

    
