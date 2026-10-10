"""
Physical cloth grasp by pressing and closing (experiment 006).
The fingertips press the cloth near a corner and close,
so that friction gathers the cloth into a ridge between the fingers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corner_tracking"))
from cloth_corner_tracker import detect_corner_vertex_indices, get_corner_world_positions

TASK = "Isaac-Lift-Cloth-Franka"

parser = argparse.ArgumentParser(description="Press-and-close physical cloth grasp")
parser.add_argument("--press_depth", type=float, default=0.0, help="Fingertip target below the cloth height [m]")
parser.add_argument("--inset", type=float, default=0.02, help="Grasp point distance from the corner [m]")
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--close_steps", type=int, default=220)
parser.add_argument("--lift_height", type=float, default=0.060)
parser.add_argument("--lift_success_height", type=float, default=0.020)
parser.add_argument("--hold_steps", type=int, default=180)

add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

def configure_scene():
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0
    env_cfg.scene.support_neg_y = None
    env_cfg.scene.support_pos_y = None

    # Same physics as physical_grasp_cloth_corner.py (baseline)
    env_cfg.scene.table.spawn.size = (1.3, 0.5, 1.05)
    env_cfg.scene.table.spawn.visible = True
    env_cfg.scene.table.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.04,
        dynamic_friction=0.03,
        restitution=0.0,
        friction_combine_mode="min",
    )

    # [006] Place the cloth at the table center (no table edge needed)
    env_cfg.scene.deformable.init_state.pos = (0.5, 0.0, 0.012)
    env_cfg.scene.deformable.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cloth_material = env_cfg.scene.deformable.spawn.physics_material
    cloth_material.surface_thickness = 0.006
    cloth_material.density = 150.0
    cloth_material.static_friction = 2.0
    cloth_material.elasticity_damping = 0.25
    cloth_material.dynamic_friction = 1.5
    cloth_material.bend_damping = 0.25

    env_cfg.scene.robot.spawn.make_uninstanceable = True
    env_cfg.scene.robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=1.5,
        restitution=0.0,
        friction_combine_mode="max",
    )
    env_cfg.scene.robot.actuators["panda_hand"].joint_effort_limit = 70.0
    env_cfg.scene.robot.actuators["panda_hand"].stiffness =500.0
    env_cfg.scene.robot.actuators["panda_hand"].damping = 100.0

    env_cfg.commands.deformable_pose.debug_vis = False
    env_cfg.commands.deformable_pose.success_vis_asset_name = "__disabled__"
    env_cfg.events.reset_deformable = None
    env_cfg.curriculum.gravity = None
    env_cfg.curriculum.action_rate = None
    env_cfg.actions = type(env_cfg)().actions.ik

    for term_name in vars(env_cfg.terminations):
        if term_name != "time_out":
            setattr(env_cfg.terminations, term_name, None)

    env_cfg.viewer.eye = (1.05, 0.45, 0.40)
    env_cfg.viewer.lookat =  (0.45, -0.05, 0.02)
    env_cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=env_cfg.viewer.eye, lookat=env_cfg.viewer.lookat)

    return env_cfg