#!/usr/bin/env python3
"""
Smoke test for GAN integration using real dataset samples.
"""
import os
import sys
import types
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "train"))

from train.patch_training_framework import (
    PatchFrameInterpolationTrainer,
    PatchTrainingConfig,
    PatchTrainingScheduleConfig,
    ColleaguePatchDataset,
)


def _load_yaml_config() -> dict:
    config_path = REPO_ROOT / "configs" / "colleague_training_config.yaml"
    return yaml.safe_load(config_path.read_text())


def _build_patch_config(patch_cfg: dict) -> PatchTrainingConfig:
    cfg = PatchTrainingConfig()
    for key, value in patch_cfg.items():
        setattr(cfg, key, value)
    return cfg


def _build_gan_trainer() -> PatchFrameInterpolationTrainer:
    base_cfg = _load_yaml_config()

    model_config = base_cfg.get("network", {})
    training_config = base_cfg.get("training", {}).copy()
    training_config.setdefault("log_dir", "./logs/test_gan")

    patch_config = _build_patch_config(base_cfg.get("patch", {}))
    schedule_config = PatchTrainingScheduleConfig()

    full_config = base_cfg.copy()
    gan_cfg = full_config.setdefault("gan", {})
    gan_cfg.update(
        {
            "enable": True,
            "warmup_epochs": 0,
            "lambda_adv_start": 1e-3,
            "lambda_adv_end": 1e-3,
            "lambda_adv_warmup_epochs": 0,
            "discriminator": {
                "input_channels": 4,
                "base_channels": 32,
                "num_layers": 3,
                "norm_type": "instance",
            },
            "optimizer": {
                "lr": 2e-4,
                "betas": [0.5, 0.999],
            },
            "training": {
                "d_steps": 1,
                "g_steps": 1,
            },
        }
    )

    vis_cfg = full_config.setdefault("visualization", {})
    vis_cfg["enable_visualization"] = False

    trainer = PatchFrameInterpolationTrainer(
        model_config=model_config,
        training_config=training_config,
        patch_config=patch_config,
        schedule_config=schedule_config,
        teacher_model_path=None,
        full_config=full_config,
    )
    return trainer


def _prepare_manual_optim(trainer: PatchFrameInterpolationTrainer):
    optim_cfg = trainer.configure_optimizers()
    if isinstance(optim_cfg, tuple):
        optimizers, _ = optim_cfg
    elif isinstance(optim_cfg, list):
        optimizers = optim_cfg
    elif isinstance(optim_cfg, dict):
        optimizers = [optim_cfg["optimizer"]]
    else:
        optimizers = optim_cfg

    if isinstance(optimizers, (list, tuple)):
        opt_g, opt_d = optimizers
    else:
        opt_g, opt_d = optimizers

    dummy_trainer = types.SimpleNamespace(
        optimizers=[opt_g, opt_d],
        callback_metrics={},
        current_epoch=0,
        barebones=False,
    )
    trainer.trainer = dummy_trainer

    trainer.log = types.MethodType(lambda self, *a, **k: None, trainer)
    trainer.log_dict = types.MethodType(lambda self, *a, **k: None, trainer)

    trainer.optimizers = types.MethodType(lambda self: (opt_g, opt_d), trainer)
    trainer.manual_backward = types.MethodType(
        lambda self, loss, retain_graph=False: loss.backward(retain_graph=retain_graph),
        trainer,
    )

    return opt_g, opt_d


def _real_batch(cfg: dict, patch_config: PatchTrainingConfig):
    data_cfg = cfg.get("data", {})
    hdr_cfg = cfg.get("hdr_processing", {})

    adapter_kwargs = dict(
        enable_linear_preprocessing=hdr_cfg.get("enable_linear_preprocessing", True),
        enable_srgb_linear=hdr_cfg.get("enable_srgb_linear", True),
        scale_factor=hdr_cfg.get("scale_factor", 0.70),
        tone_mapping_for_display=hdr_cfg.get("tone_mapping_for_display", "reinhard"),
        gamma=hdr_cfg.get("gamma", 2.2),
        exposure=hdr_cfg.get("exposure", 1.0),
        adaptive_exposure=hdr_cfg.get("adaptive_exposure", {"enable": False}),
        mulaw_mu=hdr_cfg.get("mulaw_mu", 500.0),
    )

    dataset = ColleaguePatchDataset(
        data_root=data_cfg.get("data_root", "./data"),
        split="train",
        patch_config=patch_config,
        adapter_kwargs=adapter_kwargs,
    )
    sample = dataset[0]
    batch = {
        "patch_input": sample["patch_input"].unsqueeze(0),
        "patch_target_residual": sample["patch_target_residual"].unsqueeze(0),
        "patch_target_rgb": sample["patch_target_rgb"].unsqueeze(0),
    }
    return batch


def test_gan_training_step_runs_without_error():
    os.environ.setdefault("TRAINING_FRAMEWORK_IMPORT_WARN", "0")
    torch.manual_seed(42)

    config_yaml = _load_yaml_config()
    trainer = _build_gan_trainer()
    trainer.train()

    opt_g, opt_d = _prepare_manual_optim(trainer)
    assert isinstance(opt_g, torch.optim.Optimizer)
    assert isinstance(opt_d, torch.optim.Optimizer)

    trainer.on_train_epoch_start()
    assert trainer.gan_enabled
    assert trainer.gan_step_enabled
    assert trainer.current_lambda_adv > 0.0

    patch_config = _build_patch_config(config_yaml.get("patch", {}))
    batch = _real_batch(config_yaml, patch_config)

    output = trainer.training_step(batch, batch_idx=0)
    assert "loss" in output
    loss_val = output["loss"]
    assert torch.isfinite(loss_val).all()
    assert trainer.training_stats["recent_losses"]
    assert trainer.current_lambda_adv > 0


if __name__ == "__main__":
    test_gan_training_step_runs_without_error()
    print("GAN integration smoke test passed.")
