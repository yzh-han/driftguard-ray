"""ResNet-18 architecture outline with layer notes."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Callable, List, Tuple, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

from driftguard_ray.model.moe import Gate, GateArgs 

# Stem: 7x7 conv (stride 2) + BN + ReLU, then 3x3 maxpool (stride 2).
# Output channels: 64, spatial downsampled to 1/4 of input.
# [B, 3, H, W] -> [B, 64, H/4, W/4]

def group_norm(dim: int) -> nn.Module:
    return nn.GroupNorm(num_groups=1, num_channels=dim)
    
    
@dataclass
class CRstStemArgs:
    in_channels: int = 3
    out_channels: int = 64
    pretrained: bool = True
    norm: Callable = group_norm # or nn.BatchNorm2d
    def __post_init__(self):
        self.base = resnet18(weights='IMAGENET1K_V1' if self.pretrained else None)
class CRstStem(nn.Module):
    """
    Stem block for ResNet, used to connect input channels to the first block.
    
    Uses ResNet18's initial layers for feature extraction consistency.
    """
    
    def __init__(self, args: CRstStemArgs):
        super().__init__()
        self.conv1 = copy.deepcopy(args.base.conv1)
        self.norm = args.norm(args.out_channels)
        self.relu = copy.deepcopy(args.base.relu)
        self.maxpool = copy.deepcopy(args.base.maxpool)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) # [B, 64, H/2, W/2]
        x = self.norm(x)    
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, H/4, W/4]
        return x

# Layer1: 2x BasicBlock, each 3x3 conv + BN + ReLU, residual add.
# Channels 64 -> 64, spatial size unchanged.

# Layer2: 2x BasicBlock, first block downsamples (stride 2).
# Channels 64 -> 128, spatial size /2.

# Layer3: 2x BasicBlock, first block downsamples (stride 2).
# Channels 128 -> 256, spatial size /2.

# Layer4: 2x BasicBlock, first block downsamples (stride 2).
# Channels 256 -> 512, spatial size /2.

# Layer: 
@dataclass
class CRstResidualArgs:
    """
    Configuration for CRstResidualAdd.
    """
    layers: List[int]
    num_sha_exp: int = 3
    topk: int = 1 
    in_dim: int = 64
    norm: Callable = group_norm

class CRstResidual(nn.Module):
    def __init__(self, args: CRstResidualArgs):
        super().__init__()
        self._out_dim = 256
        self.layer1 = self._make_layer(args.in_dim, 64, args.layers[0], stride=1, num_sha_exp=args.num_sha_exp, topk=args.topk)
        self.layer2 = self._make_layer(64, 128, args.layers[1], stride=2, num_sha_exp=args.num_sha_exp, topk=args.topk)
        self.layer3 = self._make_layer(128, 256, args.layers[2], stride=2, num_sha_exp=args.num_sha_exp, topk=args.topk)
        self.layer4 = None
        if len(args.layers) > 3:
            self.layer4 = self._make_layer(256, 512, args.layers[3], stride=2, num_sha_exp=args.num_sha_exp, topk=args.topk)
            self._out_dim = 512
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_gates = []
        l2_gates = []
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer is None: continue
            for block in layer:
                x, l1_w, l2_w = block(x)
                l1_gates.append(l1_w)
                l2_gates.append(l2_w)
        
        l1_gates = torch.stack(l1_gates, dim=1)  # [B, layers, 2]
        l2_gates = torch.stack(l2_gates, dim=1)  #  [B, layers, num_sha_exp]
        return x, l1_gates, l2_gates
    
    def _make_layer(self, in_dim: int, out_dim: int, blocks: int, stride: int, num_sha_exp: int, topk: int) -> nn.Sequential:
        layers = []
        layers.append(
            CRstBlock(
                CRstBlockArgs(
                    l1_gate_args=GateArgs(
                        in_dim=in_dim, hidden_dim=max(16, in_dim // 4), num_exp=2, topk=2
                    ),
                    l2_gate_args=GateArgs(
                        in_dim=in_dim, hidden_dim=max(16, in_dim // 4), num_exp=num_sha_exp, topk=topk
                    ),
                    exp_args=CRstExpArgs(in_dim=in_dim, out_dim=out_dim, stride=stride),
                    shortcut_args=CRstShortcutArgs(
                        in_dim=in_dim, out_dim=out_dim, stride=stride, norm=group_norm
                    ),
                )
            )
        )
        for _ in range(1, blocks):
            layers.append(
                CRstBlock(
                    CRstBlockArgs(
                        l1_gate_args=GateArgs(
                            in_dim=out_dim, hidden_dim=max(16, out_dim // 4), num_exp=2, topk=2
                        ),
                        l2_gate_args=GateArgs(
                            in_dim=out_dim, hidden_dim=max(16, out_dim // 4), num_exp=num_sha_exp, topk=1
                        ),
                        exp_args=CRstExpArgs(in_dim=out_dim, out_dim=out_dim, stride=1),
                        shortcut_args=CRstShortcutArgs(
                            in_dim=out_dim, out_dim=out_dim, stride=1
                        ),
                    )
                )
            )
        return nn.Sequential(*layers)
    

# Layer - Block: CRstBlock[Gate, CRstExp, CRstShortcut]
@dataclass
class CRstBlockArgs:
    """
    Configuration for CRstBlock.
    """
    l1_gate_args: GateArgs
    l2_gate_args: GateArgs
    exp_args: CRstExpArgs
    shortcut_args: CRstShortcutArgs

    def __post_init__(self):
        assert self.l1_gate_args.num_exp == 2, "L1 gate should have 2 exps"
class CRstBlock(nn.Module):
    """
    Basic block for ResNet, used in shallow networks like ResNet-8.
    
    This is a simplified version of the standard ResNet BasicBlock with
    batch normalization and ReLU activation.
    """
    
    def __init__(self, args: CRstBlockArgs):
        super().__init__()
        
        # Main path
        self.l1_gate = Gate(args.l1_gate_args)
        self.l2_gate = Gate(args.l2_gate_args)
        # Branch - experts
        self.shared = nn.ModuleList([CRstExp(args.exp_args) for _ in range(args.l2_gate_args.num_exp)])
        self.local = CRstExp(args.exp_args)
        # Residual connection
        self.shortcut = CRstShortcut(args.shortcut_args)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        # gate weights
        x_gate = F.adaptive_avg_pool2d(x, 1).flatten(1) # BCHW->BC
        l1_w = self.l1_gate(x_gate)  # [B, 2]
        l2_w = self.l2_gate(x_gate)  # [B, num_exp_shared]

        # x_shared - l2
        x_shared = torch.stack(
            [exp(x) for exp in self.shared], dim=1
        )  # -> [B, num_exp, C, H, W]
        x_shared *= l2_w.reshape(*l2_w.shape, 1, 1, 1)  # [B, num_exp_shared,1,1,1]
        x_shared = x_shared.sum(dim=1)  # [B, C, H, W]

        # x_local
        x_local = self.local(x)  # [B, C, H, W]

        # combine - l1
        x_out = torch.stack([x_shared, x_local], dim=1)  # [B,2,C,H,W]
        x_out *= l1_w.reshape(*l1_w.shape, 1, 1, 1)  # [B, 2,1,1,1]
        x_out = x_out.sum(dim=1)  # [B, C, H, W]

        x_out += self.shortcut(x)
        x_out = F.relu(x_out)

        return x_out, l1_w, l2_w
    
# Layer - Block - Components: CRstExp, CRstShortcut
@dataclass
class CRstExpArgs:
    """
    Configuration for CRstExpert block.
    """
    in_dim: int
    out_dim: int
    stride: int = 1
    norm: Callable = group_norm
class CRstExp(nn.Module):
    """
    Scaled BasicBlock for MoE experts.
    
    This block supports scaling down the number of channels to reduce
    computational cost while maintaining representational capacity.
    """
    
    def __init__(self, args: CRstExpArgs):
        super().__init__()
        
        self.conv1 = nn.Conv2d(args.in_dim, args.out_dim, kernel_size=3, stride=args.stride, padding=1, bias=False)
        self.bn1 = args.norm(args.out_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(args.out_dim, args.out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = args.norm(args.out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out
# Layer - Block - Components - Shortcut: CRstShortcut
@dataclass
class CRstShortcutArgs:
    """
    Configuration for CRstShortcut.
    """
    in_dim: int
    out_dim: int
    stride: int = 1
    norm:Callable = group_norm
class CRstShortcut(nn.Module):
    """
    Shortcut connection for ResNet blocks.
    """
    
    def __init__(self, args: CRstShortcutArgs):
        super().__init__()
        self.shortcut = nn.Sequential()
        if args.stride != 1 or args.in_dim != args.out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(args.in_dim, args.out_dim, kernel_size=1, stride=args.stride, bias=False),
                args.norm(args.out_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x)
    


# Head: global average pool to 1x1, then fully-connected to num_classes.
class CRstHead(nn.Module):
    """
    Head block for CResNet, used to map features to class logits.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        x = self.fc(x)  # [B, num_classes]
        return x

class CResNet(nn.Module):
    """
    Complete CResNet model combining Stem, Residual layers, and Head.
    """
    def __init__(self, stem: CRstStem, residual: CRstResidual, head: CRstHead):
        super().__init__()
        self.stem = stem
        self.residual = residual
        self.head = head
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x, l1_gates, l2_gates = self.residual(x)
        x = self.head(x)
        return x, l1_gates, l2_gates

def initialize_shared_experts_from_resnet18(
    residual: CRstResidual,
    *,
    pretrained: bool = True,
    noise_std: float = 1e-3,
) -> None:
    """Initialize shared experts from a ResNet-18 backbone with optional noise.

    Args:
        residual: Residual trunk to initialize in-place.
        pretrained: Whether to load ImageNet pretrained weights.
        noise_std: Standard deviation of Gaussian noise added to expert weights.
    """

    weights = "IMAGENET1K_V1" if pretrained else None
    base = resnet18(weights=weights)
    res_layers = [base.layer1, base.layer2, base.layer3, base.layer4]
    cr_layers = [residual.layer1, residual.layer2, residual.layer3, residual.layer4]

    def _copy_expert_weights(expert: CRstExp, block: BasicBlock) -> None:
        if expert.conv1.weight.shape == block.conv1.weight.shape:
            expert.conv1.weight.copy_(block.conv1.weight)
        elif expert.conv1.weight.shape == block.conv2.weight.shape:
            expert.conv1.weight.copy_(block.conv2.weight)

        if expert.conv2.weight.shape == block.conv2.weight.shape:
            expert.conv2.weight.copy_(block.conv2.weight)
        elif expert.conv2.weight.shape == block.conv1.weight.shape:
            expert.conv2.weight.copy_(block.conv1.weight)

        if noise_std > 0:
            expert.conv1.weight.add_(noise_std * torch.randn_like(expert.conv1.weight))
            expert.conv2.weight.add_(noise_std * torch.randn_like(expert.conv2.weight))

    with torch.no_grad():
        for cr_layer, res_layer in zip(cr_layers, res_layers, strict=False):
            if cr_layer is None:
                continue
            for cr_block, res_block in zip(cr_layer, res_layer, strict=False):
                res_block = cast(BasicBlock, res_block)
                for expert in cr_block.shared:
                    _copy_expert_weights(expert, res_block)

def get_cresnet(
    num_classes: int = 10,
    layers: List[int] = [2, 2, 1, 1], # [2, 2, 1, 1] for 18, [1,1,1] for 8
    num_sha_exp: int = 3,
    topk: int = 1,
    norm: Callable = group_norm,
    rst_wgts: bool = True,
) -> CResNet:
    """  
    Create a CResNet-8 model for lightweight federated learning.
    """
    stem_args = CRstStemArgs(norm=norm)
    residual_args = CRstResidualArgs(
        layers=layers, num_sha_exp=num_sha_exp, topk=topk, norm=norm
    )
    
    stem = CRstStem(stem_args)
    residual = CRstResidual(residual_args)
    head = CRstHead(in_dim=residual._out_dim, num_classes=num_classes)
    model = CResNet(stem, residual, head)
    if rst_wgts:
        initialize_shared_experts_from_resnet18(model.residual, pretrained=True, noise_std=1e-3)

    return model


