from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn

from driftguard_ray.model.c_vit.activation import DropPath

# Tensor 的方法
# transpose 只2维交换, permute是多维交换
# reshape 改变形状, flatten展平


# Layer1: Input Processing
# Patch Embedding
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

# Layer 2: Transformer Encoder, mha + mlp + encoder block + encoder
# mha
class MHAArgs(NamedTuple):
    embed_dim: int
    num_heads: int = 8
    bias: bool = False
    scale: Optional[float] = None
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

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
        self.attn_dropout = nn.Dropout(args.attn_dropout)
        self.proj = nn.Linear(args.embed_dim, args.embed_dim)
        self.proj_dropout = nn.Dropout(args.proj_dropout)

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
        attn = self.attn_dropout(attn)

        x = attn @ v  # B, num_heads, N, head_dim
        x = x.transpose(1, 2).reshape(B, N, C)  # B, N, embed_dim

        x = self.proj(x)  # B, NumberOfPatch+1 cls, embed_dim
        x = self.proj_dropout(x)
        return x

# mlp
class MLPArgs(NamedTuple):
    in_dim: int
    hidden_dim: Optional[int] = None
    out_dim: Optional[int] = None
    activation_layer: Callable[..., nn.Module] = nn.GELU
    dropout: float = 0.0

class MLP(nn.Module):
    def __init__(self, args: MLPArgs) -> None:
        super().__init__()

        self.fc1 = nn.Linear(args.in_dim, args.hidden_dim or args.in_dim * 4)
        self.activation = args.activation_layer()
        self.fc2 = nn.Linear(
            args.hidden_dim or args.in_dim * 4, args.out_dim or args.in_dim
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# encoder block
@dataclass
class EncoderBlockArgs:
    mha_args: MHAArgs
    mlp_args: MLPArgs
    embed_dim: int  # 输入embed_dim 的维度
    drop_path: float = 0.
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm

class EncoderBlock(nn.Module):
    def __init__(
        self,
        args: EncoderBlockArgs,
    ):
        super().__init__()
        self.norm1 = args.norm_layer(args.embed_dim)
        self.mha = MultiHeadAttention(args.mha_args)
        self.drop_path = (
            DropPath(args.drop_path) if args.drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = args.norm_layer(args.embed_dim)
        self.mlp = MLP(args.mlp_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mha(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# encoder
@dataclass
class EncoderArgs:
    depth: int
    block_args: EncoderBlockArgs  # using created layer args
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
        self.layers = nn.ModuleList(
            [EncoderBlock(layer_args) for layer_args in args.layers_args]
        )
        self.norm = args.block_args.norm_layer(args.block_args.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

# Layer 3: Classification Head
# representation
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

# head
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
    # ViT model
    pos_dropout: float = 0.1

    # Shared Arguments
    embed_dim: int = 768

    # Patch Embedding
    embed_image_size: int = 224
    embed_in_channels: int = 3
    embed_patch_size: int = 16

    # Transformer Encoder
    # encoder
    encoder_depth: int = 12
    encoder_drop_path_max: float = 0.1
    # encoder block
    eb_norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    # mlp
    mlp_hidden_dim: Optional[int] = None # default to in_dim *4
    mlp_activation_layer: Callable[..., nn.Module] = nn.GELU
    mlp_dropout: float = 0.1
    # mha
    mha_num_heads: int = 12
    mha_bias: bool = False # for qkv linear layers
    mha_scale: float | None = None  # default head_dim ** -0.5
    mha_attn_dropout: float = 0.1
    mha_proj_dropout: float = 0.1
    # Classification Head
    # representation
    repr_dim: Optional[int] = None  # if None, no representation layer
    repr_activation_layer: Callable[..., nn.Module] = nn.Tanh
    # head
    head_num_classes: int = 10

    def __post_init__(self):
        self.patch_embedding_args: PatchEmbeddingArgs = PatchEmbeddingArgs(
            image_size=self.embed_image_size,
            in_channels=self.embed_in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.embed_patch_size
        )
        # Encoder
        mha_args = MHAArgs(
            embed_dim=self.embed_dim,
            num_heads=self.mha_num_heads,
            bias=self.mha_bias,
            scale=self.mha_scale,
            attn_dropout=self.mha_attn_dropout,
            proj_dropout=self.mha_proj_dropout,
        )
        mlp_args = MLPArgs(
            in_dim=self.embed_dim,
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            activation_layer=self.mlp_activation_layer,
            dropout=self.mlp_dropout,
        )
        encoder_block_args = EncoderBlockArgs(
            mha_args=mha_args,
            mlp_args=mlp_args,
            embed_dim=self.embed_dim,
            norm_layer=self.eb_norm_layer,
        )
        self.encoder_args: EncoderArgs = EncoderArgs(
            depth=self.encoder_depth,
            block_args=encoder_block_args,
            drop_path_max=self.encoder_drop_path_max,
        )
        # Classification Head
        self.representation_args: Optional[RepresentationArgs] = (
            RepresentationArgs(
                embed_dim=self.embed_dim,
                repr_dim=self.repr_dim,
                activation_layer=self.repr_activation_layer,
            )
            if self.repr_dim is not None
            else None
        )
        self.head_args: HeadArgs = HeadArgs(
            embed_dim=self.embed_dim, num_classes=self.head_num_classes
        )


class VisonTransformer(nn.Module):
    def __init__(self, args: ViTArgs):
        super().__init__()
        # Layer 1 Input Processing
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
        self.pos_drop = nn.Dropout(args.pos_dropout)

        # Layer 2: Transformer Encoder
        self.encoder = Encoder(args.encoder_args)

        # Layer 3: Classification Head
        self.representation = (
            Representation(args.representation_args)
            if args.representation_args is not None
            else None
        )
        self.head = Head(args.head_args)


        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1 input processing
        x = self.patch_embed(x)  # BCHW -> B, T(num_patches), D(embed_dim)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # 1, 1, D, ->  B, 1, D(embed_dim)
        x = torch.concat((cls_token, x), dim=1)  # B, 1+T, D
        x = x + self.pos_embed  # B, 1+T, D
        x = self.pos_drop(x)

        # 2 transformer encoder
        x = self.encoder(x)  # B, 1+T, D

        # 3 Classification head
        x = x[:, 0]  # B, D
        if self.representation:  # optional
            x = self.representation(x)  # B, D -> B, repr_dim

        x = self.head(x)  # B, num_classes
        return x


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


def vit_base_patch16_224(
) -> VisonTransformer:
    """ViT-Base model (ViT-B/16) from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".

    Args:
        **kwargs: Additional keyword arguments for the ViT model.

    Returns:
        ViT-Base model.
    """
    args = ViTArgs(
        embed_dim=768,
        embed_image_size=224,
        embed_patch_size=16,
        encoder_depth=12,
        mha_num_heads=12,
        repr_dim=768,
        head_num_classes=10,
    )
    model = VisonTransformer(args)
    model.apply(_init_vit_weights)
    return model


if __name__ == "__main__":
    # model = vit_base_patch16_224()
    # img = torch.randn(1, 3, 224, 224)
    # out = model(img)
    # print(out.shape)  # Expected output: torch.Size([1, 196, 768])
    # print(nn.Softmax(dim=1)(out))

    model = vit_base_patch16_224()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
