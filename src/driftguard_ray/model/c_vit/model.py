
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Tuple, cast

import torch
import torch.nn as nn

from driftguard_ray import data
from driftguard_ray.model.c_vit.activation import DropPath
from driftguard_ray.model.moe import Gate, GateArgs
import torch.nn.functional as F

# Tensor 的方法
# transpose 只2维交换, permute是多维交换
# reshape 改变形状, flatten展平


# Layer1: Input Processing
# Patch Embedding: split image into patches and project to token embeddings.
@dataclass
class PatchEmbeddingArgs:
    """
    Arguments for the Patch Embedding layer.
    """

    image_size: int  # assumed square 224.
    in_channels: int  # usually 3 for RGB images.

    embed_dim: int  # usually 768 for ViT-Base.

    patch_size: int  # assumed square 16.
    kernel_size: Optional[int] = None  # usually equal to patch_size.
    stride: Optional[int] = None  # usually equal to patch_size.

    norm_layer: Callable[..., nn.Module] = nn.Identity  # optional
    activation_layer: Callable[..., nn.Module] = nn.ReLU  # optional

    def __post_init__(self):
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
class PatchEmbedding(nn.Module):
    # Conv2d with stride=patch_size serves as patchify + linear projection.
    def __init__(self, args: PatchEmbeddingArgs):
        super().__init__()
        self.img_size = args.image_size
        self.patch_size = args.patch_size
        self.num_patches = (args.image_size // args.patch_size) ** 2  # 196
        self.proj = nn.Conv2d(
            args.in_channels,
            args.embed_dim,
            kernel_size=args.kernel_size or args.patch_size,
            stride=args.stride or args.patch_size,
        )  # B, 3, 224, 224 -> B, 768, 14, 14
        self.norm = args.norm_layer(args.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size and W == self.img_size,
            "Input image size doesn't match model.",
        )
        x = self.proj(x)  # B, 3, 224, 224 -> B, 768, 14, 14
        x = x.flatten(2)  # B, 768, 14, 14 -> B, 768, 196
        x = x.transpose(1, 2)  # B, 768, 196 -> B, 196, 768
        x = self.norm(x)

        return x

# Layer 2: Transformer Encoder[Block [mha + mlp]]
# Encoder: stack of encoder blocks with final LayerNorm.

# Layer: Encoder [Block[mha + mlp] x depth] 
@dataclass
class EncoderArgs:
    depth: int # 深度
    block_args: BlockArgs  # using created layer args
    drop_path_max: Optional[float] = 0.1

    def __post_init__(self):
        self.layers_args = [self.block_args] * self.depth
        if self.drop_path_max:
            depth_drop_path = [
                x.item() for x in torch.linspace(0, self.drop_path_max, self.depth)
            ]
            for layer_args, drop_path in zip(self.layers_args, depth_drop_path):
                layer_args.drop_path = drop_path
class Encoder(nn.Module):
    def __init__(self, args: EncoderArgs):
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [Block(layer_args) for layer_args in args.layers_args]
        )
        self.norm = args.block_args.norm_layer(args.block_args.embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1_gates = []
        l2_gates = []

        for layer in self.layers:
            x, l1_w, l2_w = layer(x)
            l1_gates.append(l1_w)
            l2_gates.append(l2_w)
        x = self.norm(x)

        l1_gates = torch.stack(l1_gates, dim=1)  # [B, layers, 2]
        l2_gates = torch.stack(l2_gates, dim=1)  #  [B, layers, num_sha_exp]
        
        return x, l1_gates, l2_gates
    
# Layer - Block : Block [MHA + MLP]
@dataclass
class BlockArgs:
    mha_args: MHAArgs
    mlp_args: MLPArgs
    embed_dim: int  # 输入embed_dim 的维度
    drop_path: float = 0. # 不自己设置
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm
class Block(nn.Module):
    def __init__(
        self,
        args: BlockArgs,
    ):
        super().__init__()
        self.norm1 = args.norm_layer(args.embed_dim)
        self.mha: MultiHeadAttention = MultiHeadAttention(args.mha_args)
        self.drop_path = (
            DropPath(args.drop_path) if args.drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = args.norm_layer(args.embed_dim)
        self.mlp = MLP(args.mlp_args)

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.drop_path(self.mha(self.norm1(x)))

        mlp_out, l1_w, l2_w = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)
        return x, l1_w, l2_w

# Layer - Block - Component: Multi-Head Attention
@dataclass
class MHAArgs():
    embed_dim: int
    num_heads: int = 8
    bias: bool = True
    scale: Optional[float] = None
    _attn_dropout: float = 0.1
    _proj_dropout: float = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        args: MHAArgs,
    ):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.head_dim = args.embed_dim // args.num_heads
        torch._assert(
            self.head_dim * args.num_heads == self.embed_dim,
            "embed_dim must be divisible by num_heads",
        )
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(args.embed_dim, args.embed_dim * 3, bias=args.bias)
        self._attn_dropout = nn.Dropout(args._attn_dropout)
        self.proj = nn.Linear(args.embed_dim, args.embed_dim)
        self._proj_dropout = nn.Dropout(args._proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # B, N = patch size, C=embedding dimension
        qkv: torch.Tensor = self.qkv(x)  # B, N, 3*embed_dim
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, self.head_dim
        )  # B, N, 3, num_heads, head_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3(q,k,v), B, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, num_heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, N
        attn = attn.softmax(dim=-1)  # B, num_heads, N, N
        attn = self._attn_dropout(attn)

        x = attn @ v  # B, num_heads, N, head_dim
        x = x.transpose(1, 2).reshape(B, N, C)  # B, N, embed_dim

        x = self.proj(x)  # B, NumberOfPatch+1 cls, embed_dim
        x = self._proj_dropout(x)
        return x

# Layer - Block - Component - MLP[Exp]
@dataclass
class MLPArgs:
    l1_gate_args: GateArgs
    l2_gate_args: GateArgs
    exp_args: ExpArgs

    def __post_init__(self):
        assert self.l1_gate_args.num_exp == 2, "L1 gate should have 2 exps"
class MLP(nn.Module):
    def __init__(self, args: MLPArgs) -> None:
        super().__init__()
        # Main path
        self.l1_gate = Gate(args.l1_gate_args)
        self.l2_gate = Gate(args.l2_gate_args)
        # Branch - experts
        self.shared = nn.ModuleList([Exp(args.exp_args) for _ in range(args.l2_gate_args.num_exp)])
        self.local = Exp(args.exp_args)

    def forward(self, x: torch.Tensor) :
         # gate weights
        x_cls = x[:, 0] # B, D
        l1_w = self.l1_gate(x_cls)  # [B, 2]
        l2_w = self.l2_gate(x_cls)  # [B, num_exp_shared]

        # x_shared - l2
        x_shared = torch.stack(
            [exp(x) for exp in self.shared], dim=1
        )  # -> [B, num_exp, T, D]
        x_shared *= l2_w.reshape(*l2_w.shape, 1, 1)  # [B, num_exp_shared,1,1]
        x_shared = x_shared.sum(dim=1)  # [B, T, D]

        # x_local
        x_local = self.local(x)  # [B, T, D]

        # combine - l1
        x_out = torch.stack([x_shared, x_local], dim=1)  # [B, 2, T, D]
        x_out *= l1_w.reshape(*l1_w.shape, 1, 1)  # [B, 2,1,1]
        x_out = x_out.sum(dim=1)  # [B, T, D]

        x_out = F.gelu(x_out)

        return x_out, l1_w, l2_w
    
# Layer - Block - Component - Expert: Exp
class ExpArgs(NamedTuple):
    in_dim: int
    hidden_dim: Optional[int] = None
    out_dim: Optional[int] = None
    activation_layer: Callable[..., nn.Module] = nn.GELU
    dropout: float = 0.1
class Exp(nn.Module):
    def __init__(self, args: ExpArgs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(args.in_dim, args.hidden_dim or args.in_dim * 2)
        self.activation = args.activation_layer()
        self.fc2 = nn.Linear(
            args.hidden_dim or args.in_dim * 2, args.out_dim or args.in_dim
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Layer 3: Classification Head
# Representation: optional projection before final classifier.
class RepresentationArgs(NamedTuple):
    embed_dim: int
    repr_dim: Optional[int]
    activation_layer: Callable[..., nn.Module] = nn.Tanh

class Representation(nn.Module):
    def __init__(self, args: RepresentationArgs):
        super().__init__()
        self.fc = (
            nn.Linear(args.embed_dim, args.repr_dim) if args.repr_dim else None
        )
        self.activation = args.activation_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc:
            x = self.fc(x)
            x = self.activation(x)
        return x

# Head: linear classifier on the CLS token.
class HeadArgs(NamedTuple):
    embed_dim: int
    num_classes: int

class Head(nn.Module):
    def __init__(self, args: HeadArgs):
        super().__init__()
        self.fc = nn.Linear(args.embed_dim, args.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)




# ViT model
@dataclass
class ViTArgs:
    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 16
    # head
    num_classes: int = 10
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    repr_dim: Optional[int] = None

    _pos_dropout: float = 0.1

    def __post_init__(self):
        self.patch_embedding_args: PatchEmbeddingArgs = PatchEmbeddingArgs(
            image_size=self.image_size,
            in_channels=self.in_channels,
            embed_dim=self.dim,
            patch_size=self.patch_size
        )
        self.encoder_args: EncoderArgs = EncoderArgs(
            depth=self.depth,
            block_args=BlockArgs(
                mha_args=MHAArgs(
                    embed_dim=self.dim,
                    num_heads=self.num_heads,
                    bias=False,
                ),
                mlp_args=MLPArgs(
                    l1_gate_args=GateArgs(
                        in_dim=self.dim,
                        num_exp=2,
                        topk=2,
                    ),
                    l2_gate_args=GateArgs(
                        in_dim=self.dim,
                        num_exp=3,
                        topk=1,
                    ),
                    exp_args=ExpArgs(
                        in_dim=self.dim,
                    ),
                ),
                embed_dim=self.dim,
            ),
        )
        # Classification Head
        self.representation_args: Optional[RepresentationArgs] = (
            RepresentationArgs(
                embed_dim=self.dim,
                repr_dim=self.repr_dim,
            )
            if self.repr_dim is not None
            else None
        )
        self.head_args: HeadArgs = HeadArgs(
            embed_dim=self.dim, num_classes=self.num_classes
        )


class VisonTransformer(nn.Module):
    def __init__(self, args: ViTArgs):
        super().__init__()
        # Layer 1 Input Processing: patch embedding + CLS token + position encoding.
        self.patch_embed = PatchEmbedding(args.patch_embedding_args)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, args.patch_embedding_args.embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                (
                    args.patch_embedding_args.image_size
                    // args.patch_embedding_args.patch_size
                )
                ** 2
                + 1,
                args.patch_embedding_args.embed_dim,
            )
        )
        self.pos_drop = nn.Dropout(args._pos_dropout)

        # Layer 2: Transformer Encoder: stacked blocks over token sequence.
        self.encoder = Encoder(args.encoder_args)

        # Layer 3: Classification Head: CLS token -> optional repr -> logits.
        self.representation = (
            Representation(args.representation_args)
            if args.representation_args is not None
            else None
        )
        self.head = Head(args.head_args)


        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1 input processing
        x = self.patch_embed(x)  # BCHW -> B, T(num_patches), D(embed_dim)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # 1, 1, D, ->  B, 1, D(embed_dim) 每个B 一个 cls
        x = torch.concat((cls_token, x), dim=1)  # B, 1+T, D
        x = x + self.pos_embed  # B, 1+T, D
        x = self.pos_drop(x)

        # 2 transformer encoder
        x, l1_gates, l2_gates = self.encoder(x)  # B, 1+T, D

        # 3 Classification head
        x = x[:, 0]  # B, D
        if self.representation:  # optional
            x = self.representation(x)  # B, D -> B, repr_dim

        x = self.head(x)  # B, num_classes
        return x, l1_gates, l2_gates


def _init_vit_weights(m: nn.Module) -> None:
    """ViT weight initialization"""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def load_timm_vit_partial(
    model: VisonTransformer,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Load patch embedding, CLS/pos embeddings, and MHA weights from timm ViT.

    Args:
        model: Target CViT model to receive weights.
        state_dict: Pretrained state dict (e.g., from timm ViT).

    Returns:
        Mapping with lists of loaded and skipped keys.
    """
    loaded: list[str] = []
    skipped: list[str] = []

    def _copy_param(target: torch.Tensor, key: str) -> None:
        if key not in state_dict:
            skipped.append(key)
            return
        src = state_dict[key]
        if target.shape != src.shape:
            skipped.append(key)
            return
        with torch.no_grad():
            target.copy_(src)
        loaded.append(key)

    # Patch embedding + CLS/pos embeddings.
    _copy_param(
        cast(torch.Tensor, model.patch_embed.proj.weight),
        "patch_embed.proj.weight",
    )
    if model.patch_embed.proj.bias is not None:
        _copy_param(
            cast(torch.Tensor, model.patch_embed.proj.bias),
            "patch_embed.proj.bias",
        )
    _copy_param(cast(torch.Tensor, model.cls_token), "cls_token")
    _copy_param(cast(torch.Tensor, model.pos_embed), "pos_embed")

    # MHA weights per block.
    for idx, block in enumerate(model.encoder.layers):
        block = cast(Block, block)
        _copy_param(
            cast(torch.Tensor, block.mha.qkv.weight),
            f"blocks.{idx}.attn.qkv.weight",
        )
        if block.mha.qkv.bias is not None:
            _copy_param(
                cast(torch.Tensor, block.mha.qkv.bias),
                f"blocks.{idx}.attn.qkv.bias",
            )
        _copy_param(
            cast(torch.Tensor, block.mha.proj.weight),
            f"blocks.{idx}.attn.proj.weight",
        )
        _copy_param(
            cast(torch.Tensor, block.mha.proj.bias),
            f"blocks.{idx}.attn.proj.bias",
        )

    return {"loaded": loaded, "skipped": skipped}


def load_timm_vit_partial_from_model(
    model: VisonTransformer,
    timm_name: str = "vit_tiny_patch16_224",
    pretrained: bool = True,
) -> dict[str, list[str]]:
    """Load partial weights from a timm ViT model.

    Args:
        model: Target CViT model to receive weights.
        timm_name: timm model name (e.g., "vit_tiny_patch16_224").
        pretrained: Whether to load pretrained timm weights.

    Returns:
        Mapping with lists of loaded and skipped keys.

    Raises:
        ImportError: If timm is not installed.
    """
    try:
        import timm  # type: ignore
    except ImportError as exc:
        raise ImportError("timm is required to load timm ViT weights.") from exc

    timm_model = timm.create_model(timm_name, pretrained=pretrained)
    state_dict = timm_model.state_dict()
    return load_timm_vit_partial(model, state_dict)


# 参数量12,707,014: dim=192, depth=12, num_heads=3,
def get_cvit(
    num_classes: int = 10,
    image_size: int = 224,
    in_channels: int = 3,
    patch_size: int = 16,
    dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    ) -> VisonTransformer:
    """
    Create a CViT model instance.
    """
    args = ViTArgs(
        image_size=image_size,
        in_channels=in_channels,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
    )
    # Modify MLPArgs for CViT
    model = VisonTransformer(args)
    model.apply(_init_vit_weights)
    if args.image_size == 224:
        load_timm_vit_partial_from_model(model)
    return model
