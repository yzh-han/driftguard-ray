"""Synchronization helpers for federated client/server coordination."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from driftguard_ray.federate.observation import Observation
from driftguard_ray.federate.params import FedParam, Params
from driftguard_ray.federate.server.cluster import GroupState
from driftguard_ray.federate.server.state import ReqState, RetrainState
from driftguard_ray.config import get_logger
logger = get_logger("server_sync")

@dataclass
class ServerSyncCoordinator:
    """Coordinate server-side synchronization barriers for each phase."""

    req_state: ReqState

    _step_lock = threading.Lock()
    _step_cv = threading.Condition(_step_lock)

    _obs_lock = threading.Lock()
    _obs_cv = threading.Condition(_obs_lock)

    _trig_lock = threading.Lock()
    _trig_cv = threading.Condition(_trig_lock)

    def await_adv_step(self, cid: int, on_step: Callable) -> None:
        """
        Await acknowledgment for advancing the step from a client.
        """
        with self._step_lock:
            self.req_state.step[cid].recv = True
            if self.req_state.all_recv("step"):
                logger.info(f"clients:{list(self.req_state.step.keys())}")
                logger.debug("All [step_req] received from clients.")
                on_step()
                

                self.req_state.reset()
                logger.info(f"clients after reset:{[f'{c}: {r.recv}' for c, r in self.req_state.step.items()]}")

                self._step_cv.notify_all()
            else:

                self._step_cv.wait()

    def await_upload_obs(
        self,
        cid: int,
        on_obs: Callable,
        obs: Observation,
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        """
        Await acknowledgment for uploading observations from a client.
        """
        with self._obs_lock:
            self.req_state.obs[cid].recv = True
            self.req_state.obs[cid].payload = obs

            if self.req_state.all_recv("obs"):
                logger.debug("All [obs_req] received from clients.")
                obs_list = [
                    self.req_state.obs[c].payload for c in self.req_state.obs.keys()
                ]
                on_obs(obs_list, grp_state, rt_state) # update cluster etc.
                
                self.req_state.reset()
                self._obs_cv.notify_all()
            else:
                self._obs_cv.wait()
            

    def await_trig(
        self,
        cid: int,
        on_trig: Callable[[List[Observation], List[FedParam], RetrainState, GroupState, FedParam]],
        obs: Observation,
        fed_params: FedParam,
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """
        Await acknowledgment for triggering an action from a client.
        """
        with self._trig_lock:
            self.req_state.trig[cid].recv = True
            self.req_state.trig[cid].payload = obs, fed_params

            if self.req_state.all_recv("trig"):
                logger.debug("All [trig_req] received from clients.")
                payloads = [
                    self.req_state.trig[c].payload for c in self.req_state.trig.keys()
                ]
                obs_list, fed_params_list = zip(*payloads)  # type: ignore
                # to be implemented
                on_trig(obs_list, fed_params_list, rt_state, grp_state, param_state)
                
                self.req_state.reset()
                self._trig_cv.notify_all()
            else:
                self._trig_cv.wait()