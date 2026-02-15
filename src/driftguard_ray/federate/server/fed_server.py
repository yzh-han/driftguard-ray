from dataclasses import dataclass
from typing import List

from driftguard_ray.federate.observation import Observation
from driftguard_ray.federate.params import FedParam
from driftguard_ray.federate.server.state import ReqState, RetrainState
from driftguard_ray.federate.server.retrain_strategy import (
    RetrainStrategy,
)
from driftguard_ray.config import get_logger
from driftguard_ray.federate.server.sync import ServerSyncCoordinator
from driftguard_ray.federate.server.cluster import ClusterArgs, GroupState
from driftguard_ray.protocol.types import (
    ReqAdvStepArgs,
    ReqAdvStepRes,
    ReqTrigArgs,
    ReqTrigRes,
    ReqUploadObsArgs,
    ReqUploadObsRes,
)
from driftguard_ray.runtime.interfaces import DataServiceEndpoint

logger = get_logger("fed_server")

@dataclass
class FedServerArgs:
    """Arguments for constructing a federated server.

    Attributes:
        data_endpoint: Data service endpoint for stop chaining.
        num_clients: Total number of clients in the federation.
        retrain_rounds: Default number of rounds for retraining.
        retrain_strategy: Optional strategy to handle retraining triggers.
    """

    data_endpoint: DataServiceEndpoint | None
    num_clients: int
    rt_round: int # default communication rounds
    retrain_strategy: RetrainStrategy
    clu_args: ClusterArgs
    

class FedServer:
    def __init__(
        self,
        args: FedServerArgs,
    ):
        # STATES
        self.param_state: FedParam = FedParam()
        self.grp_state: GroupState = GroupState(args.num_clients, args.clu_args)
        self.rt_state: RetrainState = RetrainState(_rt_round=args.rt_round)

        self.rt_strategy: RetrainStrategy = args.retrain_strategy
        
        # runtime
        self._time_step: int = 0
        self._data_endpoint: DataServiceEndpoint | None = args.data_endpoint
        self._sync: ServerSyncCoordinator = ServerSyncCoordinator(
            ReqState(args.num_clients)
        )

        logger.info(f"FedServer starting ...")

    def stop(self) -> bool:
        """Request federated server stop.

        Returns:
            True if shutdown was triggered.
        """
        if self._data_endpoint is not None:
            self._data_endpoint.stop()
        logger.info("Shutting down the FedServer...")
        return True

    def req_adv_step(self, args: ReqAdvStepArgs) -> ReqAdvStepRes:
        (cid,) = args

        # perform once
        def on_step():
            self._time_step += 1
            logger.info(f"[Time step] advanced to [{self._time_step}]")

        self._sync.await_adv_step(cid, on_step)
        return (self._time_step,)

    def req_upload_obs(self, args: ReqUploadObsArgs) -> ReqUploadObsRes:
        """Return group parameters, or empty if no groups."""
        cid, obs = args
        # store observation

        def on_obs(obs_list: List[Observation], grp_state: GroupState, rt_state: RetrainState) -> None:
            self.rt_strategy.on_obs(obs_list, grp_state, rt_state)
            logger.info(f"-* [Ave Acc] *-  {sum(o.accuracy for o in obs_list)/len(obs_list):.4f}")
            logger.info(
                f"[-* Groups *-] {[f'{g}: {acc:.2f}' for g, acc in Observation.group_ave_acc(obs_list, grp_state.groups)]}"
            )
        self._sync.await_upload_obs(cid, on_obs, obs, self.grp_state, self.rt_state)
        
        fed_params = FedParam()
        # if self.grp_state.groups: # has groups
        #     if len(self.grp_state.get_group(cid).params) == FedParam.LOCAL_SIZE:
        #         fed_params.local = self.grp_state.get_group(cid).params
        #     else:
        #         fed_params = FedParam.separate(self.grp_state.get_group(cid).params)

        return fed_params,

    def req_trig(self, args: ReqTrigArgs) -> ReqTrigRes:
        cid, obs, fed_params = args
        
        self.current_fed_params_list: List[FedParam] = []
        def on_trig(
            obs_list: List[Observation],
            fed_params_list: List[FedParam],
            rt_state: RetrainState,
            grp_state: GroupState,
            param_state: FedParam,
        ) -> None:
            # 闭包 获取所有客户端的 fed_params
            self.current_fed_params_list.extend(fed_params_list)

            self.rt_strategy.on_trig(obs_list, fed_params_list, rt_state, grp_state, param_state)
            logger.info(f"[Retrain Trigger] rt_cfg: {rt_state.rt_cfg}, retrain: {rt_state.remain_round}")

        self._sync.await_trig(cid, on_trig, obs, fed_params, self.rt_state, self.grp_state, self.param_state)
        
        assert len(self.current_fed_params_list) == self.grp_state._num_clients, (
            "Mismatch in collected fed_params."
        )
        res_fed_params, rt_cfg = self.rt_strategy.res_trig(
            cid,
            self.rt_state,
            self.param_state,
            self.grp_state,
            self.current_fed_params_list,
            )
       
        return res_fed_params, rt_cfg  # placeholder
