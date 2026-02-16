from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Tuple

from driftguard_ray import data
from driftguard_ray.config import get_logger
from driftguard_ray.federate.observation import Observation
from driftguard_ray.federate.params import FedParam, ParamType, Params, aggregate_params
from driftguard_ray.federate.retrain_config import RetrainConfig
from driftguard_ray.federate.server.cluster import Group, GroupState
from driftguard_ray.federate.server.state import RetrainState
    
from statistics import mean

logger = get_logger("retrain_strategy")


@dataclass
class RetrainStrategy(ABC):
    """Pluggable retraining strategy for trigger and aggregation behavior."""
    data_port:int = 11001
    server_port:int = 11002
    name: str=""

    thr_sha_acc_pct: float | None = None
    cluster_thr: float | None = None
    min_group_size: int | None = None

    @abstractmethod
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        """
        Handle uploaded observations before retraining decisions.
        """

    @abstractmethod
    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """
        Handle retraining trigger and aggregation logic.
        """

    @abstractmethod
    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        pass

@dataclass
class Never(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    name: str = "never"
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
        logger.debug("Retraining never triggered.")

    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.NONE,
        ], "Never only supports NONE."
        fed_params = FedParam()
        return fed_params, rt_state.rt_cfg
    
@dataclass
class AveTrig(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    thr_acc: float = 0.65
    name: str = "average"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        ave_acc = mean([obs.accuracy for obs in obs_list])
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if ave_acc < self.thr_acc:
                # - 全局重训练
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.FULL)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
        else:
            param_state.gate = aggregate_params(
                [p.gate for p in fed_params_list],
            )
            param_state.local = aggregate_params(
                [p.local for p in fed_params_list],
            )
            param_state.other = aggregate_params(
                [p.other for p in fed_params_list],
            )
            if rt_state.stage == RetrainState.Stage.ONGOING:
                rt_state.remain_round -= 1
            elif rt_state.stage == RetrainState.Stage.COMPLETED:
                rt_state.rt_cfg.trigger = False
                logger.debug("Retraining ended.")
            else:
                raise ValueError("Inconsistent retrain state.")
    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.FULL,
            ParamType.NONE,
        ], "AveTrig only supports FULL, NONE."
        # blank for no retrain
        fed_params = FedParam()
        if not rt_state.rt_cfg.trigger and rt_state.rt_cfg.param_type == ParamType.NONE:
            return fed_params, rt_state.rt_cfg
        
        # retrain
        if cid in rt_state.rt_cfg.selection:
            fed_params.gate = param_state.gate or fed_params_list[cid].gate
            fed_params.local = param_state.local or fed_params_list[cid].local
            fed_params.other = param_state.other or fed_params_list[cid].other
        return fed_params, rt_state.rt_cfg
    
@dataclass
class PerCTrig(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    thr_acc: float = 0.65
    name: str = "per_client"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        drop_clients = [
            c for c in range(len(obs_list)) if obs_list[c].accuracy < self.thr_acc
        ]
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if drop_clients:
                # True, Drop clients, FULL
                rt_state.rt_cfg = RetrainConfig(True, drop_clients, ParamType.FULL)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
        else:
            param_state.gate = aggregate_params(
                [fed_params_list[c].gate for c in rt_state.rt_cfg.selection],
            )
            param_state.local = aggregate_params(
                [fed_params_list[c].local for c in rt_state.rt_cfg.selection],
            )
            param_state.other = aggregate_params(
                [fed_params_list[c].other for c in rt_state.rt_cfg.selection],
            )
            if rt_state.stage == RetrainState.Stage.ONGOING:
                rt_state.remain_round -= 1
            elif rt_state.stage == RetrainState.Stage.COMPLETED:
                rt_state.rt_cfg.trigger = False
                logger.debug("Retraining ended.")
            else:
                raise ValueError("Inconsistent retrain state.")
    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.FULL,
            ParamType.NONE,
        ], "PerCTrig only supports FULL, NONE."
        # blank for no retrain
        fed_params = FedParam()
        if not rt_state.rt_cfg.trigger and rt_state.rt_cfg.param_type == ParamType.NONE:
            return fed_params, rt_state.rt_cfg
        
        # retrain 
        if cid in rt_state.rt_cfg.selection:
            fed_params.gate = param_state.gate or fed_params_list[cid].gate
            fed_params.local = param_state.local or fed_params_list[cid].local
            fed_params.other = param_state.other or fed_params_list[cid].other
        return fed_params, rt_state.rt_cfg
@dataclass
class MoEAve(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    thr_acc: float = 0.65

    name: str = "moe_ave"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        ave_acc = mean([obs.accuracy for obs in obs_list])
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if ave_acc < self.thr_acc:
                # - 全局重训练
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.MOE)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
        else:
            param_state.gate = aggregate_params(
                [p.gate for p in fed_params_list],
            )
            param_state.other = aggregate_params(
                [p.other for p in fed_params_list],
            )

            if rt_state.stage == RetrainState.Stage.ONGOING:
                rt_state.remain_round -= 1
            elif rt_state.stage == RetrainState.Stage.COMPLETED:
                rt_state.rt_cfg.trigger = False
                logger.debug("Retraining ended.")
            else:
                raise ValueError("Inconsistent retrain state.")
    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.MOE,
            ParamType.NONE,
        ], "MoEAve only supports MOE, NONE."
        # blank for no retrain
        fed_params = FedParam()
        if not rt_state.rt_cfg.trigger and rt_state.rt_cfg.param_type == ParamType.NONE:
            return fed_params, rt_state.rt_cfg
        
        # retrain 
        # retrain local for selected clients
        if cid in rt_state.rt_cfg.selection:
            fed_params.gate = param_state.gate or fed_params_list[cid].gate
            fed_params.other = param_state.other or fed_params_list[cid].other
        return fed_params, rt_state.rt_cfg
@dataclass
class MoEPerC(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    thr_acc: float = 0.65

    name: str = "moe_perC"

    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        drop_clients = [
            c for c in range(len(obs_list)) if obs_list[c].accuracy < self.thr_acc
        ]
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if drop_clients:
                # True, Drop clients, FULL
                rt_state.rt_cfg = RetrainConfig(True, drop_clients, ParamType.MOE)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
        else:
            param_state.gate = aggregate_params(
                [
                    fed_params_list[c].gate
                    for c in rt_state.rt_cfg.selection
                ],
            )
            param_state.other = aggregate_params(
                [
                    fed_params_list[c].other
                    for c in rt_state.rt_cfg.selection
                ],
            )
            
            if rt_state.stage == RetrainState.Stage.ONGOING:
                rt_state.remain_round -= 1
            elif rt_state.stage == RetrainState.Stage.COMPLETED:
                rt_state.rt_cfg.trigger = False
                logger.debug("Retraining ended.")

    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.MOE,
            ParamType.NONE,
        ], "MoEPerC only supports MOE, NONE."
        # blank for no retrain
        fed_params = FedParam()
        if not rt_state.rt_cfg.trigger and rt_state.rt_cfg.param_type == ParamType.NONE:
            return fed_params, rt_state.rt_cfg
        
        # retrain 
        # retrain local for selected clients
        if cid in rt_state.rt_cfg.selection:
            fed_params.gate = param_state.gate or fed_params_list[cid].gate
            fed_params.other = param_state.other or fed_params_list[cid].other

        return fed_params, rt_state.rt_cfg  

@dataclass
class Cluster(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    thr_acc: float = 0.65
    name: str = "cluster"
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        rt_state.is_cluster = True

        fps = [obs.fingerprint for obs in obs_list]
        # AgglomerativeClustering requires at least 2 samples.
        if len(fps) >= 2:
            grp_state.update(fps)
        logger.info(f"[Updated groups]: {grp_state.groups}")

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """Apply the default retraining logic to the current round.

        Args:
            context: Runtime retrain context to mutate.
            obs_list: Observations from all clients for the round.
            params_list: Parameters uploaded by all clients.

        Returns:
            None.
        """
        group_accs = Observation.group_ave_acc(obs_list, grp_state.groups)
        grps = [g for g, acc in group_accs if acc < self.thr_acc]

        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if grps:
                # - 分组重训练
                rt_state.rt_cfg = RetrainConfig(
                    True, [c for g in grps for c in g.clients], ParamType.CLUSTER
                )
                rt_state.remain_round = rt_state._rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")

        else:
            grp_aggregate(
                    [FedParam.merge(p) for p in fed_params_list],
                    rt_state.rt_cfg.selection,
                    grp_state,
                )
            if rt_state.stage == RetrainState.Stage.ONGOING:
                rt_state.remain_round -= 1
            elif rt_state.stage == RetrainState.Stage.COMPLETED:
                rt_state.rt_cfg.trigger = False
                logger.debug("Retraining ended.")

    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.CLUSTER,
            ParamType.NONE,
        ], "Cluster only supports CLUSTER, NONE."

        fed_params = FedParam()
        if not rt_state.rt_cfg.trigger and rt_state.rt_cfg.param_type == ParamType.NONE:
            return fed_params, rt_state.rt_cfg

        if rt_state.rt_cfg.param_type == ParamType.CLUSTER:
            # retrain local for selected clients
            if cid in rt_state.rt_cfg.selection:
                _fed_params = fed_params_list[cid]
                if not FedParam.GATE_SIZE:
                    FedParam.GATE_SIZE = len(_fed_params.gate)
                if not FedParam.LOCAL_SIZE:
                    FedParam.LOCAL_SIZE = len(_fed_params.local)
                if not FedParam.OTHER_SIZE:
                    FedParam.OTHER_SIZE = len(_fed_params.other)
                fed_params = (
                    FedParam.separate(grp_state.get_group(cid).params) if grp_state.get_group(cid).params
                    else fed_params_list[cid]
                )
        return fed_params, rt_state.rt_cfg
@dataclass
class Driftguard(RetrainStrategy):
    """Default retraining strategy based on reliance and group accuracy."""
    thr_reliance: float = 0.1
    thr_group_acc: float = 0.65
    thr_sha_acc_pct: float = 0.95

    name: str = "driftguard"
    
    def __post_init__(self):
        self.thr_sha_acc: float = self.thr_group_acc * self.thr_sha_acc_pct

        self.act_gate: bool = False
        self.act_local: bool = False
        self.act_other: bool = False
        self.is_first_step: bool = True

    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        fps = [obs.fingerprint for obs in obs_list]
        # AgglomerativeClustering requires at least 2 samples.
        if len(fps) >= 2:
            grp_state.update(fps)
        # logger.info(f"[Updated groups]: {grp_state.groups}")

    def on_trig(
        self,
        obs_list: List[Observation],
        fed_params_list: List[FedParam],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """Apply the default retraining logic to the current round.

        Args:
            context: Runtime retrain context to mutate.
            obs_list: Observations from all clients for the round.
            fed_params_list: Parameters uploaded by all clients.

        Returns:
            None.
        """
        reliance = Observation.ave_reliance(obs_list)
        ave_acc = mean([obs.accuracy for obs in obs_list])
        group_accs = Observation.group_ave_acc(obs_list, grp_state.groups)
        grps = [g for g, acc in group_accs if acc < self.thr_group_acc]

        logger.info(f"Reliance: {reliance:.2f}, Ave Acc: {ave_acc:.2f}")

        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            # if reliance < self.thr_reliance or self.is_first_step:
            self.act_gate = False
            self.act_local = False
            self.act_other = False
            if ave_acc < self.thr_sha_acc:
                self.is_first_step = False
                self.act_other = True
                self.act_gate = True
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.DG_FULL)
                
                rt_state.remain_round = rt_state._rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
                
            if grps:
                # - 分组重训练
                self.act_local = True
                self.act_gate = True
                rt_state.rt_cfg = RetrainConfig(
                    True,
                    [c for g in grps for c in g.clients],
                    ParamType.DG_PARTIAL if not self.act_other else ParamType.DG_TOGETHER
                )
                rt_state.remain_round = rt_state._rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
        
        else:
            # 2. 继续训练
            param_state.gate = aggregate_params([p.gate for p in fed_params_list],)
            if self.act_local:
                local_params_list = [fed_params.local for fed_params in fed_params_list]
                grp_aggregate(
                    local_params_list,
                    rt_state.rt_cfg.selection,
                    grp_state,
                )
            if self.act_other:
                param_state.other = aggregate_params([p.other for p in fed_params_list])

            # 3. 更新状态
            if rt_state.stage == RetrainState.Stage.ONGOING:
                rt_state.remain_round -= 1
            elif rt_state.stage == RetrainState.Stage.COMPLETED:
                rt_state.rt_cfg.trigger = False
                logger.debug("Retraining ended.")

    def res_trig(
        self,
        cid: int,
        rt_state: RetrainState,
        param_state: FedParam,
        grp_state: GroupState,
        fed_params_list: List[FedParam],
    ) -> Tuple[FedParam, RetrainConfig]:
        "return params need to retrain, "
        assert rt_state.rt_cfg.param_type in [
            ParamType.DG_FULL,
            ParamType.DG_PARTIAL,
            ParamType.DG_TOGETHER,
            ParamType.NONE,
        ], "Driftguard only supports DG_FULL, DG_PARTIAL, and NONE."
        # blank for no retrain
        fed_params = FedParam()
        if not rt_state.rt_cfg.trigger and rt_state.rt_cfg.param_type == ParamType.NONE:
            return fed_params, rt_state.rt_cfg
        
        # retrain
        fed_params.gate = (
            param_state.gate or fed_params_list[cid].gate
        )  # always retrain gate

        if self.act_local:
            if cid in rt_state.rt_cfg.selection:
                fed_params.local = grp_state.get_group(cid).params or fed_params_list[cid].local
        if self.act_other:
            fed_params.other = param_state.other or fed_params_list[cid].other
        
        return fed_params, rt_state.rt_cfg
        
def grp_aggregate(
    params_list: List[Params], # List of client parameters
    sel_clients: List[int],
    grp_state: GroupState,
) -> None:
    # 聚合组内参数
    # 更新组内参数
    grps = grp_state.unique_groups(sel_clients)
    for g in grps:
        g.params = aggregate_params(
            [params_list[c] for c in g.clients if c in sel_clients]
        )
        