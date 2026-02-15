from dataclasses import dataclass
import json
from pathlib import Path
from time import sleep

import ray
import torch
from driftguard_ray.exp import DATASET, MODEL, Exps
from driftguard_ray.federate.server.retrain_strategy import (
    Driftguard,
    Never,
    AveTrig,
    PerCTrig,
    MoEAve,
    MoEPerC,
    Cluster,
    RetrainStrategy,
)
from driftguard_ray.federate.server.fed_server import FedServerArgs
from driftguard_ray.federate.client.client import FedClientArgs
from driftguard_ray.data.service import DataServiceArgs
from driftguard_ray.data.drift_simulation import DriftEventArgs
from driftguard_ray.model.training.trainer import TrainConfig, Trainer
from driftguard_ray.recorder import Recorder
from driftguard_ray.runtime.interfaces import DataServiceEndpoint, ServerEndpoint
from driftguard_ray.runtime.ray.actors import (
    RayDataServiceActor,
    RayDataServiceEndpoint,
    RayFedClientActor,
    RayFedServerActor,
    RayServerEndpoint,
)
from driftguard_ray.config import get_logger
from driftguard_ray.federate.server.cluster import ClusterArgs

logger = get_logger("launch")


@dataclass
class LaunchConfig:
    """Configuration for the local federated launch."""

    exp_root: str
    exp_name: str

    # data service
    sample_size_per_step: int
    dataset: DATASET

    # client
    total_steps: int
    batch_size: int
    device: str
    model: MODEL
    num_clients: int
    epochs: int
    lr: float

    # server
    rt_round: int
    strategy: RetrainStrategy
    cluster_thr: float = 0.2
    min_group_size: int = 3
    w_size: int = 3

    # ports
    data_port: int = 12099
    server_port: int = 12000

    seed: int = 42


def build_client_args(
    cid: int,
    cfg: LaunchConfig,
    data_endpoint: DataServiceEndpoint,
    server_endpoint: ServerEndpoint,
    resource: dict[str, float] | None = None,
) -> FedClientArgs:
    """Construct FedClient args for a single client actor."""

    args = FedClientArgs(
        cid=cid,
        data_endpoint=data_endpoint,
        server_endpoint=server_endpoint,
        trainer=Trainer(
            cfg.model.fn(cfg.dataset.num_classes),
            config=TrainConfig(
                epochs=cfg.epochs,
                device=cfg.device,
                lr=cfg.lr,
                accumulate_steps=1,
                early_stop=True,
                cp_name=f"{cfg.dataset.name}-{cfg.model.value}",
            ),
        ),
        total_steps=cfg.total_steps,
        batch_size=cfg.batch_size,
        img_size=cfg.dataset.img_size,
        exp_name=cfg.exp_name,
        exp_root=cfg.exp_root,
        resource=resource,
    )
    return args


#######################################
# Main Launching Code
#######################################
exps = Exps(
    datasets=[
        DATASET.DG5,
        # DATASET.PACS,
        # DATASET.DDN
    ],
    models=[
        MODEL.CRST_S,
        # MODEL.CVIT_S,
        # MODEL.CRST_M,
        # MODEL.CVIT
    ],
    strategies=[
        # Never(),
        AveTrig(thr_acc=0.85, data_port=13101, server_port=13102),
        # PerCTrig(thr_acc=0.85, data_port=13201, server_port=13202),
        # MoEAve(thr_acc=0.85, data_port=13301, server_port=13302),
        # MoEPerC(thr_acc=0.85, data_port=14401, server_port=14402),
        # Cluster(thr_acc=0.85, data_port=13501, server_port=13502),
        # Driftguard(thr_group_acc=0.85, thr_sha_acc_pct=0.95, data_port=14601, server_port=14602),
    ],
    device="cuda:0" if torch.cuda.is_available() else "cpu",  # <--------------------
).exps


def main() -> None:
    """Start the local data service, server, and clients."""
    for exp in exps:
        print("\n\n")
        logger.info(
            f"[Experiment]: {exp.name}, Dataset: {exp.dataset.name}, Model: {exp.model.value}, Strategy: {exp.strategy.name}, lr: {exp.lr}"
        )
        # cluster_thr, min_group_size = 0.12, 2 # <--------------------
        clustr = (
            str(exp.cluster_thr).split(".")[0] + str(exp.cluster_thr).split(".")[-1]
        )
        cfg = LaunchConfig(
            # exp_root=f"exp/ablation_{exp.strategy.name}",
            # exp_root=f"exp/{exp.strategy.name}_clu{clustr}_mgsize{min_group_size}",
            # exp_root="exp/main_acc60",
            exp_root=f"exp/ablations/mingrp/acc{str(exp.strategy.thr_sha_acc_pct).split('.')[-1]}_clu{str(exp.strategy.cluster_thr).split('.')[-1]}_mingrp{exp.strategy.min_group_size}",
            exp_name=exp.name,
            # data service
            sample_size_per_step=30,  # <--------------------
            dataset=exp.dataset,
            # client
            total_steps=30,  # <--------------------
            batch_size=8,
            num_clients=30, # 5
            model=exp.model,
            device=exp.device,
            epochs=2,  # 20 <--------------------
            lr=exp.lr,
            # server
            rt_round=2,  # 5 communication rounds <--------------------
            strategy=exp.strategy,
            cluster_thr=exp.strategy.cluster_thr
            or exp.cluster_thr,  # 0.3,  # <--------------------
            min_group_size=exp.strategy.min_group_size or exp.min_group_size,
            data_port=exp.strategy.data_port,  # <--------------------
            server_port=exp.strategy.server_port,  # <--------------------
        )
        logger.info(f"root: {cfg.exp_root}, name: {cfg.exp_name}")

        event_args = DriftEventArgs(
            n_time_steps=cfg.total_steps,
            n_clients=cfg.num_clients,
            n_sudden=3,
            n_gradual=3,
            n_stage=1,
            aff_client_ratio_range=(0.1, 0.15),
            start=0.05,
            end=0.8,
            dist_range=(1, 3),
            gradual_duration_ratio=0.15,
            seed=cfg.seed,
        )

        #  ## data service
        data_args = DataServiceArgs(
            meta_path=cfg.dataset.path,
            num_clients=cfg.num_clients,
            sample_size=cfg.sample_size_per_step,
            drift_event_args=event_args,
            seed=cfg.seed,
        )

        data_actor = RayDataServiceActor.remote(data_args)
        server_actor = RayFedServerActor.options(max_concurrency=32).remote(
            FedServerArgs(
                data_endpoint=None,
                num_clients=cfg.num_clients,
                rt_round=cfg.rt_round,
                retrain_strategy=cfg.strategy,
                clu_args=ClusterArgs(
                    thr=cfg.cluster_thr,
                    min_group_size=cfg.min_group_size,
                    w_size=cfg.w_size,
                ),
            ),
        )
        data_ep = RayDataServiceEndpoint(data_actor)
        server_ep = RayServerEndpoint(server_actor)

        try:
            client_actors = [
                RayFedClientActor.options(
                    num_gpus=0.01 if "cuda" in cfg.device else 0,
                ).remote(
                    build_client_args(cid, cfg, data_ep, server_ep, resource = {"pi_2": 1})
                )
                for cid in range(cfg.num_clients)
            ]
            for step in range(cfg.total_steps):
                logger.info(f"=== Time step {step + 1} ===")
                ray.get([actor.step_1.remote() for actor in client_actors])
                ray.get([actor.step_2.remote() for actor in client_actors])
                ray.get([actor.step_3.remote() for actor in client_actors])

            recorders: list[Recorder] = ray.get(
                [actor.get_recorder.remote() for actor in client_actors]
            )
            for cid, recorder in enumerate(recorders):
                recorder.record(cid)
        finally:
            data_ep.stop()
            server_ep.stop()

        logger.info("Launch finished.")



if __name__ == "__main__":
    if not ray.is_initialized():
        import os
        os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] = "1"  # 放在 ray.init 前


        ray.init(
            # address=cfg.ray_address,
            address="auto",
            runtime_env={"working_dir": ".",},
            ignore_reinit_error=True,
            log_to_driver=True,
        )
        
    main()

    if ray.is_initialized():
        ray.shutdown()

# ray start --head --resources='{"my_res": 2, "ssd": 1}'
# # worker 节点同理
# ray start --address=<head-ip>:6379 --resources='{"my_res": 1}'

# a = MyActor.options(
#     num_cpus=2,
#     num_gpus=1,
#     resources={"ssd": 1, "node_elseptimo": 0.01}
# ).remote()