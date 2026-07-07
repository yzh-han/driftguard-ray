

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from driftguard_ray.recorder import Recorder
from driftguard_ray.federate.observation import Fp, Observation
from driftguard_ray.federate.params import FedParam, ParamType
from driftguard_ray.model.dataset import ListDataset, get_inference_transform, get_train_transform
from driftguard_ray.model.training.trainer import TrainConfig, Trainer
from driftguard_ray.model.utils import freeze_layer, get_trainable_params
from driftguard_ray.runtime.interfaces import DataServiceEndpoint, ServerEndpoint

from typing import Callable, Tuple, List, Dict
from driftguard_ray.config import get_logger

logger = get_logger("fedclient")

@dataclass
class FedClientArgs:
    cid: int
    data_endpoint: DataServiceEndpoint
    server_endpoint: ServerEndpoint
    model_fn: Callable[[int], nn.Module]
    num_classes: int
    train_config: TrainConfig
    total_steps: int = 20
    batch_size: int = 6

    img_size: int = 28 # 28, 224 ,224

    resource: Dict[str, float] | None = None

    exp_name: str = "exp"
    exp_root: str = "exp"

class FedClient:
    """client"""
    
    def __init__(
        self,
        args: FedClientArgs,
    ):
        self.cid = args.cid
        self.img_size: int = args.img_size
        self.data_endpoint: DataServiceEndpoint = args.data_endpoint
        self.server_endpoint: ServerEndpoint = args.server_endpoint
        self.total_steps: int = args.total_steps
        self.batch_size: int = args.batch_size
        self.model_fn: Callable[[int], nn.Module] = args.model_fn
        self.num_classes: int = args.num_classes
        self.train_config: TrainConfig = args.train_config
        self.fed_param: FedParam = self._init_fed_param()
        self._buffer: List =[]
        self.recorder = Recorder(args.exp_name, args.exp_root)

        self.resource = args.resource

        self.time_step = 1

    def _init_fed_param(self) -> FedParam:
        model = self.model_fn(self.num_classes)
        cp_path = Path("cp") / f"{self.train_config.cp_name}.pth"
        if cp_path.exists():
            model.load_state_dict(torch.load(cp_path, map_location="cpu"))
            logger.info(f"{self.cid} Loaded model weights from {cp_path}")
        return FedParam.get(model)

    def _apply_fed_param(self, fed_params: FedParam) -> None:
        if fed_params.gate:
            self.fed_param.gate = fed_params.gate
        if fed_params.local:
            self.fed_param.local = fed_params.local
        if fed_params.other:
            self.fed_param.other = fed_params.other
    
    def step_1(self):
        # step 1. inference
        self.time_step, = self.server_endpoint.req_adv_step((self.cid,))
        self.samples = self.data_endpoint.get_data((self.cid, self.time_step)) 

    def step_2(self):
        # step 2. upload observations, update local params
        self.obs = self.inference(self.samples)
        _, = self.server_endpoint.req_upload_obs((self.cid, self.obs))
        self.recorder.update_acc(self.time_step, self.obs.accuracy) 

    def step_3(self):
        # step 3. trigger retrain if needed
        # obs = self.inference(samples) #
        # self.recorder.update_acc(time_step, obs.accuracy)
        
        # train_sets, val_sets = (
        #     [*self._buffer, *self.samples[: -len(self.samples) // 3]],
        #     self.samples[-len(self.samples) // 3 :],
        # )
        
        PRE_FILL_TURN = 3
        while len(self._buffer) >= len(self.samples) * PRE_FILL_TURN:
            # 分割训练集和验证集，3:1 prefillturn : 1
            train_sets, val_sets = (
                self._buffer,
                self.samples,
            )
            
            # request 3 req_trig
            fed_params, rt_cfg = self.server_endpoint.req_trig(
                (self.cid, self.obs, self.fed_param)
            )
            
            # stop
            if not rt_cfg.trigger:
                if rt_cfg.param_type != ParamType.NONE:
                    self._apply_fed_param(fed_params) # n. last params update
                break
            
            # 1. 准备更新参数
            self._apply_fed_param(fed_params)

            # 2. no params need to retrain
            if fed_params.is_empty():
                logger.debug(f"{self.cid} No parameters to retrain, skip training.")
                continue
                            
            self.train(train_sets, val_sets, self.time_step, fed_params)

        # one step done, update buffer
        # self._buffer = self.samples[-len(self.samples) // 3:]

        self._buffer = list(
            deque([*self._buffer, *self.samples], maxlen=len(self.samples) * PRE_FILL_TURN)
        )
        
    
    def get_recorder(self) -> Recorder:
        return self.recorder

    # 不会用到
    def run(self) -> Recorder:
        """Backup lifecycle entrypoint; the Ray driver usually calls step_1/2/3 directly."""
        while self.time_step <= self.total_steps:
            self.step_1()
            self.step_2()
            self.step_3()
        return self.recorder

    def inference(self, samples: List[Tuple[bytes, int]]) -> Observation:
        num_gpus = 0.05 if "cuda" in str(self.train_config.device) else 0
        return ray.get(
            _inference.options(
                num_gpus=num_gpus,
                num_cpus=0.05,
                resources=self.resource,
            ).remote(
                self.img_size,
                self.batch_size,
                self.model_fn,
                self.num_classes,
                self.train_config,
                self.fed_param,
                samples,
            )
        )
    

    def train(
        self,
        train_sets: List[Tuple[bytes, int]],
        val_sets: List[Tuple[bytes, int]],
        time_step: int,
        train_fed_params: FedParam,
    ) -> None:
        num_gpus = 0.05 if "cuda" in str(self.train_config.device) else 0
        
        fed_param, trained_epochs, times, trainable_params = ray.get(
            _train.options(
                num_gpus=num_gpus,
                num_cpus=0.05,
                resources=self.resource,
            ).remote(
                self.img_size,
                self.batch_size,
                self.model_fn,
                self.num_classes,
                self.train_config,
                self.fed_param,
                train_fed_params,
                train_sets,
                val_sets,
            )
        )
        self.fed_param = fed_param

        self.recorder.update_cost(
            time_step,
            trainable_params,
            trained_epochs,
            times,
        )

@ray.remote
def _inference(
    img_size: int,
    batch_size: int,
    model_fn: Callable[[int], nn.Module],
    num_classes: int,
    train_config: TrainConfig,
    fed_param: FedParam,
    samples: List[Tuple[bytes, int]],
) -> Observation:
    model = model_fn(num_classes)
    fed_param.set(model)
    trainer = Trainer(model, config=train_config)
    loader = DataLoader(
        ListDataset(samples, get_inference_transform(img_size)),
        batch_size=batch_size,
        shuffle=False,
    )
    metrix, l1_w, l2_w, softs = trainer.inference(loader)
    return Observation(
        accuracy=metrix.accuracy,
        reliance=l1_w.mean(dim=[0, 1])[0].item(),
        fingerprint=Fp.build(
            out_softs=softs.cpu().numpy(),
            gate_activations=l2_w.cpu().numpy(),
            w_size=3,
        ),
    )

# @ray.remote(resources={"pi_2": 1})
@ray.remote
def _train(
    img_size: int,
    batch_size: int,
    model_fn: Callable[[int], nn.Module],
    num_classes: int,
    train_config: TrainConfig,
    fed_param: FedParam,
    train_fed_params: FedParam,
    train_sets: List[Tuple[bytes, int]],
    val_sets: List[Tuple[bytes, int]],
) -> Tuple[FedParam, int, List[float], int]:
    # print("start training on remote trainer...")
    model = model_fn(num_classes)
    fed_param.set(model)
    FedParam.unfreeze(model)
    if not train_fed_params.gate:
        freeze_layer(model, include_names=["gate"])
    if not train_fed_params.local:
        freeze_layer(model, include_names=["local"])
    if not train_fed_params.other:
        freeze_layer(model, include_names=["local", "gate"], exclude=True)

    trainable_params = get_trainable_params(model)
    if trainable_params == 0:
        logger.debug("No parameters to retrain, skip training.")
        return FedParam.get(model), 0, [], trainable_params

    trainer = Trainer(model, config=train_config)
    train_loader1, train_loader2, val_loader = (
        DataLoader(
            ListDataset(train_sets, get_inference_transform(img_size)),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            ListDataset(train_sets, get_train_transform(img_size)),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            ListDataset(val_sets, get_inference_transform(img_size)),
            batch_size=batch_size,
            shuffle=False,
        ),
    )

    # 2 stage train, origin -> 增强
    history_1 = trainer.fit(train_loader1, val_loader)
    history_2 = trainer.fit(train_loader2, val_loader)

    print(f"epoch time: {torch.mean(torch.tensor([r['time'] for r in [*history_1, *history_2]])):.2f}s")

    return (
        FedParam.get(trainer.model),
        len(history_1) + len(history_2),
        [r["time"] for r in [*history_1, *history_2]],
        trainable_params,
    )
