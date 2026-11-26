"""Behavior-clone the high-level pick policy and low-level IK policy separately.

Reuses the original PPO actor BC training approach. Input episodes are loaded
with scripts.learning.demonstration_dataset (the updated hierarchical loader).

Run from the repository root with a Python environment containing torch,
stable-baselines3, gymnasium and numpy:
    python -m scripts.learning.train_bc --data runs/teleop/physical --output runs/learning/bc_hierarchical

The saved low_action_bounds.json MUST accompany low_bc_ppo.zip during inference.
BC teaches actor weights; PPO critics are not pretrained here.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.learning.demonstration_dataset import load_episodes, split_episodes
from scripts.learning.policy_contract import (
    VERSION, HighPolicySpaces, LowPolicySpaces, XY_HIGH, XY_LOW,
    LOW_ACTION_SIZE, normalize_low_action,
)


def low_action_bounds(train_episodes, position_margin_m=0.05):
    """Calculate root-frame XYZ bounds using TRAINING episodes only.

    Quaternion components and binary gripper have fixed [-1, 1] bounds.
    Stored bounds must be reused unchanged for rollout/PPO decoding.
    """
    commands = np.concatenate([e.low_actions for e in train_episodes], axis=0)
    if commands.ndim != 2 or commands.shape[1] != LOW_ACTION_SIZE or not np.isfinite(commands).all():
        raise ValueError("Expected finite low-level IK actions (steps, 8)")
    if position_margin_m <= 0:
        raise ValueError("position_margin_m must be positive")
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, +1.0, dtype=np.float32)
    lower[:3] = commands[:, :3].min(axis=0) - position_margin_m
    upper[:3] = commands[:, :3].max(axis=0) + position_margin_m
    if np.any(np.abs(commands[:, 3:7]) > 1.0001):
        raise ValueError("Recorded quaternion components exceed [-1, 1]")
    if np.any(np.abs(commands[:, 7]) > 1.0001):
        raise ValueError("Recorded gripper command exceeds [-1, 1]")
    if np.any(np.linalg.norm(commands[:, 3:7], axis=1) < 1e-6):
        raise ValueError("Recorded IK command contains zero quaternion")
    return lower, upper


def _arrays(episodes, level):
    if level == "high":
        obs = [e.observations for e in episodes]
        acts = [e.actions for e in episodes]
    elif level == "low":
        obs = [e.low_observations for e in episodes]
        acts = [e.low_actions for e in episodes]
    else:
        raise ValueError("level must be high or low")
    x, y = np.concatenate(obs), np.concatenate(acts)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training data includes NaN or infinite values")
    return x.astype(np.float32), y.astype(np.float32)


def _normalized_low(actions, lower, upper):
    # Explicitly reject out-of-workspace data rather than silently clipping labels.
    return np.stack([normalize_low_action(a, lower, upper) for a in actions])


def _make_model(level, seed, device):
    env = HighPolicySpaces() if level == "high" else LowPolicySpaces()
    return PPO(
        "MlpPolicy", env, seed=seed, device=device,
        n_steps=32, batch_size=32, learning_rate=3e-4,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                       "log_std_init": -2.5},
    )


def _validate(model, x, desired, level, low_bounds=None):
    model.policy.set_training_mode(False)
    with torch.no_grad():
        estimates = model.policy.get_distribution(torch.as_tensor(x, device=model.device)).distribution.mean
        predictions = estimates.clamp(-1.0, 1.0).cpu().numpy()
    if level == "high":
        scale = (XY_HIGH - XY_LOW) / 2
        rmse = float(np.sqrt(np.mean(((predictions - desired) * scale) ** 2)))
        return rmse, {"coordinate_rmse_m": rmse}
    lower, upper = low_bounds
    # Evaluate in physical units for XYZ and separately for rotation/gripper.
    true_raw = lower + 0.5 * (desired + 1) * (upper - lower)
    pred_raw = lower + 0.5 * (predictions + 1) * (upper - lower)
    pos_rmse = float(np.sqrt(np.mean((pred_raw[:, :3] - true_raw[:, :3]) ** 2)))
    quat_mae = float(np.mean(np.abs(pred_raw[:, 3:7] - true_raw[:, 3:7])))
    gripper_accuracy = float(np.mean((pred_raw[:, 7] >= 0) == (true_raw[:, 7] >= 0)))
    return pos_rmse, {"position_rmse_m": pos_rmse,
                      "quaternion_component_mae": quat_mae,
                      "gripper_sign_accuracy": gripper_accuracy}


def _atomic_write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _save_checkpoint(path, state):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _dataset_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def train(episodes, output, *, epochs=100, batch_size=64, seed=0, device="cpu",
          validation_fraction=0.2, position_margin_m=0.05, resume=False):
    """Train separate PPO actors using BC; keep existing 4-D BC files untouched."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_set, val_set = split_episodes(episodes, validation_fraction, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    lower, upper = low_action_bounds(train_set, position_margin_m)
    # A validation episode outside the training workspace should be reported,
    # not used to expand bounds (which would leak validation information).
    hx, hy = _arrays(train_set, "high")
    hvx, hvy = _arrays(val_set, "high")
    lx, raw_ly = _arrays(train_set, "low")
    lvx, raw_lvy = _arrays(val_set, "low")
    ly = _normalized_low(raw_ly, lower, upper)
    # Validation labels can fall outside [-1,1]. Keep the true value to
    # measure extrapolation error, but normalization is only linear here.
    lvy = (2 * (raw_lvy - lower) / (upper - lower) - 1).astype(np.float32)
    bounds = {"contract": VERSION,
              "action_frame": "robot_root",
              "action_order": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
              "low": lower.tolist(), "high": upper.tolist(),
              "position_margin_m": position_margin_m,
              "source": "training episodes only"}
    checkpoint_dir = output / "checkpoints"
    settings_path = output / "bc_training_config.json"

    settings = {
        "contract": VERSION,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "validation_fraction": validation_fraction,
        "position_margin_m": position_margin_m,
        "train_episodes": [str(e.path) for e in train_set],
        "validation_episodes": [str(e.path) for e in val_set],
        "dataset_hash": _dataset_hash(
            hx, hy, hvx, hvy, lx, ly, lvx, lvy
        ),
        "bounds": bounds,
    }

    if resume:
        if not settings_path.is_file():
            raise ValueError(
                "No compatible BC training configuration found"
            )

        saved = json.loads(settings_path.read_text())
        if saved != settings:
            raise ValueError(
                "BC dataset or training settings have changed"
            )

        print("[BC RESUME] Training settings verified", flush=True)
    else:
        if settings_path.exists() or any(
            (output / name).exists()
            for name in (
                "high_bc_ppo.zip",
                "low_bc_ppo.zip",
                "bc_training_report.json",
                "checkpoints",
            )
        ):
            raise ValueError(
                "Existing BC run. Use --resume or a new output directory"
            )

        _atomic_write(
            settings_path,
            json.dumps(settings, indent=2).encode("utf-8")
        )

    checkpoint_dir.mkdir(exist_ok=True)

    (output / "low_action_bounds.json").write_text(
        json.dumps(bounds, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {"contract": VERSION, "method": "BC using PPO actor networks",
              "train_episodes": [e.path for e in train_set],
              "validation_episodes": [e.path for e in val_set],
              "high_samples": len(hx), "low_samples": len(lx),
              "note": "BC does not train the value networks; test simulator performance separately."}
    for level, x_np, y_np, vx_np, vy_np, filename in [
        ("high", hx, hy, hvx, hvy, "high_bc_ppo"),
        ("low", lx, ly, lvx, lvy, "low_bc_ppo"),
    ]:
        model = _make_model(level, seed, device)
        model.cloth_contract = VERSION
        model.cloth_policy_level = level
        model.cloth_training_modes = sorted({e.mode for e in train_set})
        x = torch.as_tensor(x_np, device=model.device)
        y = torch.as_tensor(y_np, device=model.device)
        optimizer = model.policy.optimizer
        best = float("inf")
        history = []
        start_epoch = 0

        checkpoint = checkpoint_dir / f"{level}_latest.pt"
        best_path = output / (filename + ".zip")

        if resume and checkpoint.is_file():
            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )

            if (
                state["contract"] != VERSION
                or state["level"] != level
            ):
                raise ValueError("BC checkpoint mismatch")

            start_epoch = int(state["completed_epochs"])
            history = state["history"]
            best = float(state["best"])

            if not 1 <= start_epoch <= epochs:
                raise ValueError("Invalid checkpoint epoch")

            if len(history) != start_epoch:
                raise ValueError("Invalid checkpoint history")

            model.policy.load_state_dict(state["policy"])
            optimizer.load_state_dict(state["optimizer"])

            _atomic_write(
                best_path, state["best_model_zip"]
            )

            torch.set_rng_state(state["torch_rng"])

            if state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])

            print(
                f"[BC RESUME] {level}: "
                f"completed={start_epoch}/{epochs}",
                flush=True
            )

        for epoch in range(start_epoch, epochs):
            model.policy.set_training_mode(True)
            for indices in torch.randperm(len(x), device=model.device).split(batch_size):
                prediction = model.policy.get_distribution(x[indices]).distribution.mean
                loss = torch.nn.functional.mse_loss(prediction, y[indices])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
                optimizer.step()
            score, metrics = _validate(model, vx_np, vy_np, level,
                                       (lower, upper) if level == "low" else None)
            history.append({"epoch": epoch + 1, **metrics})
            if score < best:
                best = score
                temp_model = output / (filename + ".tmp.zip")
                model.save(temp_model)
                temp_model.replace(best_path)

            state = {
                "contract": VERSION,
                "level": level,
                "completed_epochs": epoch + 1,
                "policy": model.policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best": best,
                "history": history,
                "best_model_zip": best_path.read_bytes(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if model.device.type == "cuda"
                    else None
                ),
            }

            _save_checkpoint(checkpoint, state)

            print(
                f"[BC CHECKPOINT] {level} "
                f"epoch={epoch + 1}/{epochs}",
                flush=True
            )

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"[{level}] epoch={epoch + 1} {metrics}", flush=True)
        report[level] = {"best_validation_score": best, "history": history,
                         "checkpoint": str(output / (filename + ".zip"))}

    (output / "bc_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/learning/bc_hierarchical"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--position_margin_m", type=float, default=0.05)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume BC from the last completed epoch"
    )
    args = parser.parse_args()
    episodes, rejected = load_episodes(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rejected_episodes.json").write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"accepted={len(episodes)} rejected={len(rejected)}", flush=True)
    if not episodes:
        parser.error("No compatible successful demonstrations found. Check rejected_episodes.json")
    try:
        result = train(episodes, args.output, epochs=args.epochs,
                       batch_size=args.batch_size, validation_fraction=args.validation_fraction,
                       seed=args.seed, device=args.device,
                       position_margin_m=args.position_margin_m,
                       resume=args.resume)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"high_checkpoint": result["high"]["checkpoint"],
                      "low_checkpoint": result["low"]["checkpoint"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\\n[BC INTERRUPTED] Completed epoch checkpoints "
            "are preserved. Use --resume to continue.",
            flush=True
        )
        sys.exit(130)
