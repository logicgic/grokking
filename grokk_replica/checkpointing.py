"""Atomic checkpoint storage for single-process training."""
import os
import random
from pathlib import Path

import torch


DATASET_FIELDS = (
    "train_pairs", "val_pairs", "ordered_group_elements1",
    "ordered_group_elements2", "idx2vocab", "vocab2idx", "n_vocab", "n_out",
)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(directory, model, optimizer, scheduler, step, config, dataset):
    rng = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": config,
        "dataset": {key: getattr(dataset, key) for key in DATASET_FIELDS},
        "rng": rng,
    }
    path = Path(directory) / "latest.pt"
    atomic_save(checkpoint, path)
    print(f"Saved checkpoint at step {step}: {path}", flush=True)


def load_checkpoint(path, model, optimizer, scheduler, config, dataset):
    # Full checkpoints contain Python RNG state; load only trusted local files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use a full latest.pt checkpoint")
    previous = checkpoint["config"]
    if previous["dataset"] != config["dataset"]:
        raise ValueError("Resume dataset configuration differs from checkpoint")
    if previous["model"]["transformer_config"] != config["model"]["transformer_config"]:
        raise ValueError("Resume model configuration differs from checkpoint")
    for key in ("lr", "weight_decay", "betas", "warmup_steps", "bsize", "eval_every", "eval_batches", "num_workers"):
        if previous["train"][key] != config["train"][key]:
            raise ValueError(f"Resume train.{key} differs from checkpoint")
    step = checkpoint["step"]
    budget = config["train"]["max_steps"]
    if budget is not None and budget < step:
        raise ValueError("max_steps must be at least the checkpoint step (total budget)")
    for key in DATASET_FIELDS:
        setattr(dataset, key, checkpoint["dataset"][key])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["rng"]["python"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    cuda_rng = checkpoint["rng"].get("cuda")
    if cuda_rng is not None and torch.cuda.is_available():
        if len(cuda_rng) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(cuda_rng)
    print(f"Resumed checkpoint at step {step}: {path}", flush=True)
    return step
