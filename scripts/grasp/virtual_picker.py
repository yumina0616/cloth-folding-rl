"""Reusable virtual cloth grasp, independent of robot motion planning.

All positions are world coordinates in metres. One picker owns one environment
and one vertex. Surface cloth uses per-step state projection (an approximation,
not a physical contact grasp); volume assets use kinematic targets if available.
Call update EACH physics step, using the measured end-effector position:
before the solver for kinematic targets, after it for state projection.
The attachment preserves its initial translation offset; rotation is not tracked.
"""
from __future__ import annotations

import math
import torch


def as_tensor(value):
    """Support both older Torch buffers and newer Isaac Lab ProxyArray buffers."""
    return value if isinstance(value, torch.Tensor) else value.torch


class VirtualPicker:
    def __init__(self, cloth, anchor_position, *, env_id=0, grasp_radius=0.025):
        if not math.isfinite(grasp_radius) or grasp_radius <= 0:
            raise ValueError("grasp_radius must be positive and finite")
        self.cloth = cloth
        self.anchor_position = anchor_position
        self.env_id = env_id
        if not 0 <= env_id < as_tensor(cloth.data.nodal_pos_w).shape[0]:
            raise ValueError("env_id is outside the cloth batch")
        self.grasp_radius = grasp_radius
        self.vertex_id = None
        self._offset = None
        self._previous_target = None
        self._saved_target = None
        self.backend = (
            "kinematic" if getattr(cloth.data, "nodal_kinematic_target", None) is not None
            else "state_projection"
        )

    @property
    def is_grasping(self):
        return self.vertex_id is not None

    @property
    def target_position_w(self):
        return None if not self.is_grasping else self._point(self.anchor_position()) + self._offset

    def _point(self, value):
        nodes = as_tensor(self.cloth.data.nodal_pos_w)
        point = torch.as_tensor(value, device=nodes.device, dtype=nodes.dtype)
        if point.shape != (3,) or not bool(torch.isfinite(point).all()):
            raise ValueError("Expected a finite world position with shape (3,)")
        return point

    def _write(self, name, values):
        writer = getattr(self.cloth, name + "_index", None)
        if writer is None:
            writer = getattr(self.cloth, name)
        writer(values.contiguous(), env_ids=[self.env_id])

    def grasp(self, position_w, *, vertex_id=None):
        """Attach a nearby vertex, returning False when out of reach.

        position_w selects the desired cloth location. Both that location and
        the measured gripper must be within grasp_radius of the chosen vertex.
        An explicit vertex_id prevents accidentally selecting another corner.
        """
        if self.is_grasping:
            raise RuntimeError("Release the current vertex before grasping again")
        nodes = as_tensor(self.cloth.data.nodal_pos_w)[self.env_id]
        point = self._point(position_w)
        anchor = self._point(self.anchor_position())
        if vertex_id is None:
            vertex_id = int(torch.linalg.vector_norm(nodes - point, dim=-1).argmin())
        if not 0 <= vertex_id < nodes.shape[0]:
            raise ValueError("vertex_id is outside the cloth mesh")
        node = nodes[vertex_id]
        if not bool(torch.isfinite(node).all()):
            raise ValueError("Selected vertex is not finite")
        if max(float(torch.linalg.vector_norm(node - point)),
               float(torch.linalg.vector_norm(node - anchor))) > self.grasp_radius:
            return False
        if self.backend == "kinematic":
            target = as_tensor(self.cloth.data.nodal_kinematic_target)[self.env_id, vertex_id]
            if float(target[3]) == 0.0:
                raise RuntimeError("Selected vertex is already constrained by another owner")
            self._saved_target = target.clone()
        self.vertex_id = int(vertex_id)
        self._offset = (node - anchor).clone()
        self._previous_target = node.clone()
        self.update()
        return True

    def update(self, dt=None):
        """Refresh attachment at the backend's step boundary; dt enables velocity."""
        if not self.is_grasping:
            return
        if dt is not None and (not math.isfinite(dt) or dt <= 0):
            raise ValueError("dt must be positive and finite")
        target = self._point(self.anchor_position()) + self._offset
        if self.backend == "kinematic":
            targets = as_tensor(self.cloth.data.nodal_kinematic_target)[self.env_id:self.env_id + 1].clone()
            targets[0, self.vertex_id, :3] = target
            targets[0, self.vertex_id, 3] = 0.0
            self._write("write_nodal_kinematic_target_to_sim", targets)
        else:
            positions = as_tensor(self.cloth.data.nodal_pos_w)[self.env_id:self.env_id + 1].clone()
            velocities = as_tensor(self.cloth.data.nodal_vel_w)[self.env_id:self.env_id + 1].clone()
            positions[0, self.vertex_id] = target
            velocities[0, self.vertex_id] = 0.0 if dt is None else (target - self._previous_target) / dt
            self._write("write_nodal_pos_to_sim", positions)
            self._write("write_nodal_velocity_to_sim", velocities)
        self._previous_target = target.clone()

    def release(self):
        """Release immediately, preserving all other vertex constraints."""
        if self.is_grasping and self.backend == "kinematic":
            targets = as_tensor(self.cloth.data.nodal_kinematic_target)[self.env_id:self.env_id + 1].clone()
            targets[0, self.vertex_id] = self._saved_target
            targets[0, self.vertex_id, :3] = as_tensor(self.cloth.data.nodal_pos_w)[self.env_id, self.vertex_id]
            self._write("write_nodal_kinematic_target_to_sim", targets)
        self.vertex_id = None
        self._offset = self._previous_target = self._saved_target = None

    def reset(self):
        """Call BEFORE env.reset(), so no constraint survives an episode."""
        self.release()


def create_picker(cloth, anchor_position, *, grasp_mode="virtual", **kwargs):
    if grasp_mode != "virtual":
        raise ValueError("This module supports grasp_mode='virtual' only")
    return VirtualPicker(cloth, anchor_position, **kwargs)
