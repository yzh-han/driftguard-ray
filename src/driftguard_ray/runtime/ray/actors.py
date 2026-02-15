"""Ray actor endpoints for data service and fed server."""

from dataclasses import dataclass

import ray
from ray.actor import ActorHandle, ActorProxy

from driftguard_ray.data.service import DataService, DataServiceArgs
from driftguard_ray.federate.client.client import FedClient, FedClientArgs
from driftguard_ray.federate.server.fed_server import FedServer, FedServerArgs
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
from driftguard_ray.recorder import Recorder
from driftguard_ray.runtime.interfaces import DataServiceEndpoint, ServerEndpoint


@ray.remote
class RayDataServiceActor:
    """Ray data service actor."""

    def __init__(self, args: DataServiceArgs) -> None:
        """Initialize actor with DataService.

        Args:
            args: Data service construction args.
        """
        self._service = DataService(args)
        self._stopped = False

    def get_data(self, args: GetDataArgs) -> GetDataRes:
        """Proxy call to data service."""
        if self._stopped:
            raise RuntimeError("DataService actor already stopped.")
        return self._service.get_data(args)

    def stop(self) -> bool:
        """Mark actor as stopped."""
        self._stopped = True
        return self._service.stop()


@ray.remote
class RayFedServerActor:
    """Ray federated server actor."""

    def __init__(self, args: FedServerArgs) -> None:
        """Initialize actor with FedServer.

        Args:
            args: Fed server construction args.
        """
        # print(" - - -- - RayFedServerActor initializing...")
        self._service = FedServer(args)
        self._stopped = False

    def req_adv_step(self, args: ReqAdvStepArgs) -> ReqAdvStepRes:
        """Proxy req_adv_step."""
        if self._stopped:
            raise RuntimeError("FedServer actor already stopped.")
        return self._service.req_adv_step(args)

    def req_upload_obs(self, args: ReqUploadObsArgs) -> ReqUploadObsRes:
        """Proxy req_upload_obs."""
        if self._stopped:
            raise RuntimeError("FedServer actor already stopped.")
        return self._service.req_upload_obs(args)

    def req_trig(self, args: ReqTrigArgs) -> ReqTrigRes:
        """Proxy req_trig."""
        if self._stopped:
            raise RuntimeError("FedServer actor already stopped.")
        return self._service.req_trig(args)

    def stop(self) -> bool:
        """Stop fed server actor and chained data endpoint."""
        self._stopped = True
        self._service.stop()
        return True


@ray.remote
class RayFedClientActor:
    """Ray federated client actor."""

    def __init__(self, args: FedClientArgs) -> None:
        """Initialize actor with FedClient."""
        self._client = FedClient(args)
        # Keep preload behavior identical to the previous threaded client path.
        self._client._trainer.load()
    
    def get_cid(self) -> int:
        return self._client.cid

    def step_1(self):
        self._client.step_1()
    
    def step_2(self):
        self._client.step_2()
    
    def step_3(self):
        self._client.step_3()

    def get_recorder(self) -> Recorder:
        """Fetch recorder from client."""
        return self._client.get_recorder()

    def run(self) -> Recorder:
        """Run one full client lifecycle."""
        return self._client.run()


@dataclass
class RayDataServiceEndpoint(DataServiceEndpoint):
    """Ray-backed data endpoint."""

    actor: ActorProxy

    def get_data(self, args: GetDataArgs) -> GetDataRes:
        """Fetch data from actor."""
        return ray.get(self.actor.get_data.remote(args))

    def stop(self) -> bool:
        """Stop actor."""
        return ray.get(self.actor.stop.remote())


@dataclass
class RayServerEndpoint(ServerEndpoint):
    """Ray-backed server endpoint."""

    actor: ActorProxy

    def req_adv_step(self, args: ReqAdvStepArgs) -> ReqAdvStepRes:
        """Call req_adv_step on actor."""
        return ray.get(self.actor.req_adv_step.remote(args))

    def req_upload_obs(self, args: ReqUploadObsArgs) -> ReqUploadObsRes:
        """Call req_upload_obs on actor."""
        return ray.get(self.actor.req_upload_obs.remote(args))

    def req_trig(self, args: ReqTrigArgs) -> ReqTrigRes:
        """Call req_trig on actor."""
        return ray.get(self.actor.req_trig.remote(args))

    def stop(self) -> bool:
        """Stop actor."""
        return ray.get(self.actor.stop.remote())
