from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def _ensure_training_classes_importable() -> None:
    """Lightning checkpoints may reference training-side classes.

    We proactively import them so torch.load() can unpickle safely.
    """
    try:
        import sys

        repo = Path(__file__).resolve().parents[2]
        train_dir = repo / "train"
        if str(train_dir) not in sys.path:
            sys.path.insert(0, str(train_dir))
        # The class name matters for unpickling. Import is best-effort.
        from patch_aware_dataset import PatchTrainingConfig  # noqa: F401
    except Exception:
        return

from src.npu.networks.patch import ExtraNet, PatchNetwork, PatchNetworkV2, StudentPatchNetwork


def build_model(
    network_type: str,
    base_channels: int,
    device: str,
    input_channels: int = 7,
    output_channels: int = 3,
) -> torch.nn.Module:
    t = str(network_type).lower()
    if t in {"v2", "patchnetworkv2", "patch_network_v2"}:
        model = PatchNetworkV2(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            residual_scale_factor=1.0,
        )
    elif t in {"v1", "patchnetwork", "patch_network"}:
        model = PatchNetwork(
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=base_channels,
            residual_scale_factor=1.0,
        )
    elif t in {"student_s1", "student", "student_s1_enhanced", "student_s1_asymmetric", "s1_symmetric_bc32"}:
        variant = "s1"
        if t == "student_s1_enhanced":
            variant = "s1_attn_bc20"
        elif t == "student_s1_asymmetric":
            variant = "s1_asymmetric"
        elif t == "s1_symmetric_bc32":
            variant = "s1_symmetric_bc32"
        model = StudentPatchNetwork.from_variant(variant)
    elif t in {"student_s2", "student2"}:
        model = StudentPatchNetwork.from_variant("s2")
    elif t in {"extranet"}:
        model = ExtraNet(input_channels=input_channels, output_channels=output_channels, base_channels=base_channels)
    else:
        raise ValueError(f"Unknown network_type: {network_type}")
    return model.to(device).eval()


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj  # already a state dict
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out


def load_model_from_ckpt(
    ckpt_path: str | Path,
    network_type: str,
    base_channels: int,
    device: str,
    strict: bool = False,
    input_channels: int = 7,
    output_channels: int = 3,
) -> torch.nn.Module:
    ckpt_path = Path(ckpt_path)
    _ensure_training_classes_importable()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    sd = _strip_prefix(sd, "patch_network.")
    model = build_model(network_type, base_channels, device, input_channels, output_channels)
    model.load_state_dict(sd, strict=strict)
    return model


def load_model_from_pth(
    pth_path: str | Path,
    network_type: str,
    base_channels: int,
    device: str,
    strict: bool = True,
    input_channels: int = 7,
    output_channels: int = 3,
) -> torch.nn.Module:
    pth_path = Path(pth_path)
    obj = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    sd = _extract_state_dict(obj)
    model = build_model(network_type, base_channels, device, input_channels, output_channels)
    model.load_state_dict(sd, strict=strict)
    return model
