"""Physical cloth grasp with the same interface as the virtual picker.

The cloth is held only by finger contact and friction; nothing is attached.
grasp() just records which corner vertex the gripper is supposed to hold, so the
caller can read where that corner really is while moving and detect slip or drop.
The robot motion (approach, close, lift) stays in the caller, as for the virtual
picker. All positions are world coordinates in metres.
"""
from __future__ import annotations

import math
import torch


def as_tensor(value):
    """Support both older Torch buffers and newer Isaac Lab ProxyArray buffers."""
    return value if isinstance(value, torch.Tensor) else value.torch


def physical_grasp_xy(corner_xy, center_xy, open_width, *, pad_half=0.01, edge_margin=0.005, jaw_axis="y"):
    """Grasp point near a corner so that both open fingertips land on the cloth.

    The fingers spread only along the jaw axis (world "x" or "y"): that axis needs room
    for half the open gap plus a pad, the other axis only for the pad (008 "axis" layout).
    Assumes the cloth edges are aligned with world x and y (no cloth rotation).
    """
    if not math.isfinite(open_width) or open_width <= 0:
        raise ValueError("open_width must be positive and finite")
    if jaw_axis not in ("x", "y"):
        raise ValueError("jaw_axis must be 'x' or 'y'")
    pad_room = pad_half + edge_margin
    jaw_room = open_width / 2 + pad_room
    toward_center = torch.sign(center_xy - corner_xy)
    if bool((toward_center == 0).any()):
        raise ValueError("Cloth center must differ from the corner in both x and y")
    room = (pad_room, jaw_room) if jaw_axis == "y" else (jaw_room, pad_room)
    return corner_xy + toward_center * corner_xy.new_tensor(room)


class PhysicalPicker:
    def __init__(self, cloth, anchor_position, *, env_id=0, grasp_radius=0.06):
        if not math.isfinite(grasp_radius) or grasp_radius <= 0:
            raise ValueError("grasp_radius must be positive and finite")
        self.cloth = cloth
        self.anchor_position = anchor_position
        self.env_id = env_id
        if not 0 <= env_id < as_tensor(cloth.data.nodal_pos_w).shape[0]:
            raise ValueError("env_id is outside the cloth batch")
        self.grasp_radius = grasp_radius
        self.vertex_id = None
        self.grasp_offset = None
        self.backend = "physical"

    @property
    def is_grasping(self):
        return self.vertex_id is not None

    def _corner(self):
        return as_tensor(self.cloth.data.nodal_pos_w)[self.env_id, self.vertex_id]

    def _anchor(self):
        corner = as_tensor(self.cloth.data.nodal_pos_w)
        return torch.as_tensor(self.anchor_position(), device=corner.device, dtype=corner.dtype)

    @property
    def target_position_w(self):
        """Where the held corner really is now (not a fixed offset, unlike the virtual picker)."""
        return None if not self.is_grasping else self._corner().clone()

    def corner_offset(self):
        """Current corner position relative to the gripper (TCP)."""
        return self._corner() - self._anchor()

    def drift(self):
        """How far the corner moved relative to the gripper since the grasp [m]."""
        return float(torch.linalg.vector_norm(self.corner_offset() - self.grasp_offset))

    def grasp(self, position_w, *, vertex_id):
        """Record the held corner after the fingers have closed.

        Returns False when the gripper is not near that corner. Whether the cloth is
        really held is decided later by the caller (lift check), not here.
        """
        if self.is_grasping:
            raise RuntimeError("Release the current vertex before grasping again")
        nodes = as_tensor(self.cloth.data.nodal_pos_w)[self.env_id]
        if not 0 <= vertex_id < nodes.shape[0]:
            raise ValueError("vertex_id is outside the cloth mesh")
        if float(torch.linalg.vector_norm(nodes[vertex_id] - self._anchor())) > self.grasp_radius:
            return False
        self.vertex_id = int(vertex_id)
        self.grasp_offset = self.corner_offset().clone()
        return True

    def update(self, dt=None):
        """Nothing to enforce: the physics engine holds the cloth."""

    def release(self):
        self.vertex_id = None
        self.grasp_offset = None

    def reset(self):
        self.release()


def create_picker(cloth, anchor_position, *, grasp_mode="virtual", **kwargs):
    """Pick the grasp module by name; the virtual one needs Isaac Sim (pxr) to import."""
    if grasp_mode == "physical":
        return PhysicalPicker(cloth, anchor_position, **kwargs)
    if grasp_mode == "virtual":
        from virtual_attachment_picker import create_picker as create_virtual
        return create_virtual(cloth, anchor_position, grasp_mode="virtual", **kwargs)
    raise ValueError(f"Unknown grasp_mode: {grasp_mode}")
