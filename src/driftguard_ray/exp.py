
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List

from pathlib import Path
import torch

from driftguard_ray.federate.server.retrain_strategy import RetrainStrategy, Driftguard
from driftguard_ray.model.c_resnet.model import get_cresnet
from driftguard_ray.model.c_vit.model import get_cvit
from driftguard_ray.model.utils import get_trainable_params

class DATASET(Enum):
    """
    (meta_path, num_classes, image_size)
    """
    DG5 = (Path("~/driftguard-ray/datasets/dg5/_meta.json"), 10, 28)
    PACS = (Path("~/driftguard-ray/datasets/pacs/_meta.json"), 7, 224)
    DDN = (Path("~/driftguard-ray/datasets/drift_domain_net/_meta.json"), 7, 224)
    @property
    def path(self) -> Path:
        return self.value[0]
    @property
    def num_classes(self) -> int:
        return self.value[1]
    @property
    def img_size(self) -> int:
        return self.value[2]



class MODEL(Enum):
    """Model building functions."""
    CRST_S = "crst_s"
    CRST_M = "crst_m"
    CVIT = "cvit"
    CVIT_S = "cvit_s"
    
    @property
    def fn(self) -> Callable:
        if self == MODEL.CRST_S:
            return MODEL.cresnet_s
        elif self == MODEL.CRST_M:
            return MODEL.cresnet_m
        elif self == MODEL.CVIT:
            return MODEL.cvit
        elif self == MODEL.CVIT_S:
            return MODEL.cvit_s
        else:
            raise ValueError(f"Unknown model function: {self}")
    @staticmethod
    def cresnet_s(num_classes: int) -> Callable:
        """Build a ResNet18 model."""
        return get_cresnet(num_classes, [1,1,1])
    @staticmethod
    def cresnet_m(num_classes: int) -> Callable:
        """Build a ResNet18 model."""
        return get_cresnet(num_classes)
    @staticmethod
    def cvit(num_classes: int) -> Callable:
        """Build a Cvit model."""
        return get_cvit(num_classes)
    @staticmethod
    def cvit_s(num_classes: int) -> Callable:
        """Build a Cvit model for 28x28 inputs."""
        return get_cvit(
            num_classes,
            image_size=28,
            patch_size=4,
            dim=128,
            depth=6,
            num_heads=4,
        )

# print(get_trainable_params(MODEL.CRST_S.fn(10)))
# print(get_trainable_params(MODEL.CVIT_S.fn(10)))

@dataclass
class Exp:
    name: str
    dataset: DATASET
    model: MODEL
    strategy: RetrainStrategy
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.model == MODEL.CRST_S:
            self.lr: float = 0.001
            self.cluster_thr, self.min_group_size = 0.21, 2
        elif self.model == MODEL.CRST_M:
            self.lr: float = 0.001
            self.cluster_thr, self.min_group_size = 0.12, 2
        elif self.model == MODEL.CVIT:
            self.lr: float = 0.001
            self.cluster_thr, self.min_group_size = 0.21, 2
        elif self.model == MODEL.CVIT_S:
            self.lr: float = 0.00025
            self.cluster_thr, self.min_group_size = 0.21, 2
        else:
            raise ValueError(f"Unknown model: {self.model}")
@dataclass
class Exps:
    datasets: List[DATASET] = field(
        default_factory=lambda: [
            DATASET.DG5, 
            DATASET.PACS, 
            DATASET.DDN
        ]
    )
    models: List[MODEL] = field(
        default_factory=lambda: [
            MODEL.CRST_S,
            MODEL.CRST_M,
            MODEL.CVIT,
            MODEL.CVIT_S,
        ]
    )
    strategies: List[RetrainStrategy] = field(
        default_factory=lambda: [
            Driftguard(),
        ]
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def exps(self) -> List[Exp]:
        exp_list = []
        for dataset in self.datasets:
            for model in self.models:
                if (model == MODEL.CRST_S or model == MODEL.CVIT_S ) and (
                    dataset == DATASET.PACS or dataset == DATASET.DDN
                ):
                    continue  # skip cresnet_s on pacs and ddn
                if (model == MODEL.CVIT or model == MODEL.CRST_M) and dataset == DATASET.DG5:
                    continue  # skip cvit on dg5
                for strategy in self.strategies:
                    exp_name = f"{dataset.name}-{model.value}-{strategy.name}"
                    exp = Exp(
                        name=exp_name,
                        dataset=dataset,
                        model=model,
                        strategy=strategy,
                        device=self.device,
                    )
                    exp_list.append(exp)
        return exp_list
