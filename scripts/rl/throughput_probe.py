"""Measure how fast the cloth task runs with N parallel environments.

One num_envs per run (Isaac Sim starts once per process). Run several times:
    for n in 1 4 16 64; do ~/IsaacLab/isaaclab.sh -p scripts/rl/throughput_probe.py --num_envs $n --viz none presets=isaacsim_physx; done
Each run prints one line starting with PROBE followed by JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
import isaaclab_tasks  # noqa: F401  (registers Isaac-Lift-Cloth-Franka)


def gpu_memory_mb():
    """Used GPU memory of GPU 0 from nvidia-smi (includes PhysX, unlike torch's counter)."""
    # nvidia-smi를 실행해서 숫자 하나(MB)를 float로 돌려주기
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.splitlines()[0]

    return float(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--measure_steps", type=int, default=200)
    parser.add_argument("--cloth_size", type=float, default=0.30)
    parser.add_argument("--mesh_refinement", type=float, default=32.0)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]

    cfg, _ = resolve_task_config("Isaac-Lift-Cloth-Franka", "")
    cfg.scene.num_envs = args.num_envs
    cfg.scene.deformable.spawn.size = (args.cloth_size, args.cloth_size)
    cfg.scene.deformable.spawn.edge_refinement = args.mesh_refinement
    cfg.scene.deformable.spawn.deformable_props.self_collision = True

    with launch_simulation(cfg, args):
        env = gym.make("Isaac-Lift-Cloth-Franka", cfg=cfg)
        env.reset()
        base = env.unwrapped
        # Zero action = robot holds still; we only want the simulation cost
        action = torch.zeros(env.action_space.shape, device=base.device)

        # Warm-up is not timed: the first steps include shader compilation and memory allocation
        for _ in range(args.warmup_steps):
            env.step(action)

        start_time = time.perf_counter()
        for _ in range(args.measure_steps):
            env.step(action)
        torch.cuda.synchronize()  # wait once for queued GPU work before stopping the clock
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        steps_per_s = args.measure_steps / elapsed
        result = {
            "num_envs": args.num_envs,
            "elapsed_s": round(elapsed, 2),
            "steps_per_s": round(steps_per_s, 2),
            "env_steps_per_s": round(steps_per_s * args.num_envs, 2),
            "real_time_factor": round(args.measure_steps * base.step_dt / elapsed, 3),
            "step_dt_s": base.step_dt,
            "gpu_mem_mb": gpu_memory_mb(),
        }
        print("PROBE " + json.dumps(result), flush=True)
        env.close()


if __name__ == "__main__":
    main()