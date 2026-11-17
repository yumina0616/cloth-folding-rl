"""BC pretraining of the actual PPO actor; run as python -m scripts.learning.train_bc."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.learning.demonstration_dataset import load_episodes, split_episodes
from scripts.learning.policy_contract import PolicySpaces, VERSION, XY_HIGH, XY_LOW


def train(episodes, output, *, epochs=100, batch_size=64, seed=0, device="cpu", validation_fraction=.2):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_set, val_set = split_episodes(episodes, validation_fraction, seed)
    torch.manual_seed(seed)
    model = PPO("MlpPolicy", PolicySpaces(), seed=seed, device=device, n_steps=32,
                batch_size=32, learning_rate=3e-4, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                                                               "log_std_init": -2.5})
    model.cloth_contract = VERSION
    model.cloth_training_modes = sorted({episode.mode for episode in train_set})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x = torch.as_tensor(np.concatenate([e.observations for e in train_set]), device=model.device)
    y = torch.as_tensor(np.concatenate([e.actions for e in train_set]), device=model.device)
    vx = torch.as_tensor(np.concatenate([e.observations for e in val_set]), device=model.device)
    vy = torch.as_tensor(np.concatenate([e.actions for e in val_set]), device=model.device)
    optimizer = model.policy.optimizer
    best, history = float("inf"), []
    scale = torch.as_tensor(np.tile((XY_HIGH - XY_LOW) / 2, 2), device=model.device)
    for epoch in range(epochs):
        model.policy.set_training_mode(True)
        order = torch.randperm(len(x), device=model.device)
        for indices in order.split(batch_size):
            prediction = model.policy.get_distribution(x[indices]).distribution.mean
            loss = torch.nn.functional.mse_loss(prediction, y[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.)
            optimizer.step()
        model.policy.set_training_mode(False)
        with torch.no_grad():
            prediction = model.policy.get_distribution(vx).distribution.mean.clamp(-1, 1)
            error = (prediction - vy) * scale
            rmse = float(error.square().mean().sqrt().cpu())
        history.append({"epoch": epoch + 1, "validation_coordinate_rmse_m": rmse})
        if rmse < best:
            best = rmse
            model.save(output)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} validation_coordinate_rmse_m={rmse:.5f}", flush=True)
    report = {"contract": VERSION, "algorithm": "BC on PPO actor", "seed": seed,
              "train_episodes": [e.path for e in train_set], "validation_episodes": [e.path for e in val_set],
              "training_modes": model.cloth_training_modes, "best_validation_coordinate_rmse_m": best,
              "history": history, "note": "Value network is untrained. Offline RMSE is not folding success rate."}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/learning/bc_ppo"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--validation_fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    episodes, rejected = load_episodes(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    print(f"accepted={len(episodes)} rejected={len(rejected)}", flush=True)
    try:
        report = train(episodes, args.output, epochs=args.epochs, batch_size=args.batch_size,
                       seed=args.seed, device=args.device, validation_fraction=args.validation_fraction)
    except ValueError as exc:
        parser.error(f"{exc}. See {args.output.with_suffix('.rejected.json')} for rejected episodes")
    print(json.dumps({"checkpoint": str(args.output.with_suffix('.zip')), "best_rmse_m": report[
        "best_validation_coordinate_rmse_m"]}), flush=True)


if __name__ == "__main__":
    main()
