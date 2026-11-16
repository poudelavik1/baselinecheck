"""Model registry. Adding a model is one `register(...)` line at the bottom of this file.

Importing this module does not import torch, so plots and reports can read the model
list (name, family, tier) on a laptop. Networks are built lazily by `build_model`.

Contract of every network: input RGB float in [0, 1], shape (N, 3, H, W); output logits
(N, 1, H, W). In train mode a network may return (logits, [auxiliary logits]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

FAMILIES = ("CNN", "Transformer", "Mamba", "crack-specific", "auto-configured", "my-existing")
# How a family is named in figures and thesis tables; the keys above stay the internal names.
FAMILY_LABELS = {"my-existing": "Own models"}


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


TIERS = ("core", "tier1", "tier2")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HALF = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    tier: str
    builder: Callable[..., Any] | None   # builder(pretrained: bool, config: dict) -> nn.Module
    label: str                           # name used in figures and tables
    env: str = "main"                    # conda env the model needs: main | scsegamba | nnunet
    external: bool = False               # trained by its own pipeline (nnU-Net), not by train.py


REGISTRY: dict[str, ModelSpec] = {}


def register(name: str, family: str, tier: str, builder: Callable[..., Any] | None, label: str | None = None,
             env: str = "main", external: bool = False) -> None:
    assert family in FAMILIES and tier in TIERS and name not in REGISTRY, name
    REGISTRY[name] = ModelSpec(name, family, tier, builder, label or name, env, external)


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def select(models: list[str] | None = None, tiers: list[str] | None = None, families: list[str] | None = None,
           env: str | None = None, include_external: bool = False) -> list[ModelSpec]:
    """Registered models filtered by name, tier, family and environment, in registry order."""
    chosen = []
    for spec in REGISTRY.values():
        if models and spec.name not in models:
            continue
        if tiers and spec.tier not in tiers:
            continue
        if families and spec.family not in families:
            continue
        if env and spec.env != env:
            continue
        if spec.external and not include_external:
            continue
        chosen.append(spec)
    return chosen


def build_model(name: str, pretrained: bool = True, config: dict[str, Any] | None = None):
    """Instantiate a registered network."""
    spec = get_spec(name)
    if spec.builder is None:
        raise RuntimeError(f"{name} is trained by its own pipeline (see scripts/nnunet/), not by train.py.")
    return spec.builder(pretrained, config or {})


# --------------------------------------------------------------------------- #
# Builders (all imports are deferred to build time)
# --------------------------------------------------------------------------- #
def normalized(network, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Wrap a network that expects normalized input so that it accepts RGB in [0, 1]."""
    import torch
    from torch import nn

    class Normalized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = network
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, x):
            return self.net((x - self.mean) / self.std)

    return Normalized()


def smp(architecture: str, encoder: str, **kwargs: Any) -> Callable[..., Any]:
    """segmentation_models_pytorch network with an ImageNet-pretrained encoder and one output channel."""
    def build(pretrained: bool, config: dict[str, Any]):
        import segmentation_models_pytorch as smp_library

        network = getattr(smp_library, architecture)(
            encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1,
            **kwargs)
        return normalized(network)
    return build


def _deep_supervision(config: dict[str, Any]) -> bool:
    return bool(config.get("recipe", {}).get("deep_supervision", False))


def hairline_unet(encoder: str = "resnet34") -> Callable[..., Any]:
    def build(pretrained: bool, config: dict[str, Any]):
        from .hairline_unet import HairlineUNet

        return HairlineUNet(encoder_name=encoder, pretrained=pretrained, deep_supervision=_deep_supervision(config))
    return build


def swin_unet(encoder: str = "swin_t") -> Callable[..., Any]:
    def build(pretrained: bool, config: dict[str, Any]):
        from .swin_unet import HairlineSwinUNet

        return HairlineSwinUNet(encoder_name=encoder, pretrained=pretrained, deep_supervision=_deep_supervision(config))
    return build


def deepcrack(pretrained: bool, config: dict[str, Any]):
    from .deepcrack import DeepCrack

    return normalized(DeepCrack(pretrained=pretrained))


def crackformer(pretrained: bool, config: dict[str, Any]):
    from .crackformer import build_crackformer

    return normalized(build_crackformer(), HALF, HALF)


def scsegamba(pretrained: bool, config: dict[str, Any]):
    from .scsegamba import build_scsegamba

    return normalized(build_scsegamba(), HALF, HALF)


# --------------------------------------------------------------------------- #
# The benchmark. One line per model: name, family, tier, builder.
# --------------------------------------------------------------------------- #
# My existing models (architectures ported unchanged)
register("hairline_unet", "my-existing", "core", hairline_unet("resnet34"), "HairlineUNet (R34)")
register("swin_unet", "my-existing", "core", swin_unet("swin_t"), "Swin-UNet (hairline)")
# Core smp models
register("unet_r34", "CNN", "core", smp("Unet", "resnet34"), "U-Net R34")
register("unet_r50", "CNN", "core", smp("Unet", "resnet50"), "U-Net R50")
register("unet_effb4", "CNN", "core", smp("Unet", "efficientnet-b4"), "U-Net EffNet-B4")
register("unetpp_r34", "CNN", "core", smp("UnetPlusPlus", "resnet34"), "UNet++ R34")
register("deeplabv3p_r34", "CNN", "core", smp("DeepLabV3Plus", "resnet34"), "DeepLabV3+ R34")
register("fpn_r34", "CNN", "core", smp("FPN", "resnet34"), "FPN R34")
register("unet_r34_scse", "CNN", "core", smp("Unet", "resnet34", decoder_attention_type="scse"), "U-Net R34 scSE")
# Tier 1
register("segformer_b2", "Transformer", "tier1", smp("Segformer", "mit_b2"), "SegFormer-B2")
register("unet_convnext_t", "CNN", "tier1", smp("Unet", "tu-convnext_tiny"), "U-Net ConvNeXt-T")
register("linknet_r34", "CNN", "tier1", smp("Linknet", "resnet34"), "LinkNet R34")
register("manet_r34", "CNN", "tier1", smp("MAnet", "resnet34"), "MA-Net R34")
register("deepcrack", "crack-specific", "tier1", deepcrack, "DeepCrack")
# Tier 2 (each isolated; a failure never blocks the others)
register("crackformer", "crack-specific", "tier2", crackformer, "CrackFormer-II")
register("scsegamba", "Mamba", "tier2", scsegamba, "SCSegamba", env="scsegamba")
register("nnunet", "auto-configured", "tier2", None, "nnU-Net v2 (2D)", env="nnunet", external=True)
