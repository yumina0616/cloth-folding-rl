"""Pick/place executor on ClothTeleop's real IK/PhysX loop, without keyboard input.

Import after Isaac Lab simulation launch. No successful grasp is fabricated:
physical grasp uses fingers and a measured lift check, virtual uses attachments.
"""
import math

import torch


class IsaacPickPlace:
    def __init__(self, session, lift_height=.12):
        self.session = session
        self.lift_height = lift_height

    @property
    def pairs(self):
        return self.session.pairs

    def corners(self):
        return self.session.corners()

    @torch.no_grad()
    def reset(self, seed=None):
        if seed is not None:
            self.session.base.seed(seed)
        self.session.reset()

    def hold(self, seconds):
        for _ in range(max(1, round(seconds / self.session.dt))):
            self.session.step(record=False)

    def move(self, goal, seconds, *, arc_height=0.):
        start = self.session.tcp().clone()
        count = max(1, round(seconds / self.session.dt))
        for step in range(1, count + 1):
            fraction = step / count
            self.session.target = torch.lerp(start, goal, fraction)
            self.session.target[2] += arc_height * math.sin(math.pi * fraction)
            self.session.step(record=False)
        error = float((self.session.tcp() - goal).norm())
        if error > .04:
            raise RuntimeError(f"IK target not reached (error {error:.3f}m)")

    @torch.no_grad()
    def execute(self, world_action):
        from physical_picker import as_tensor
        s = self.session
        picker = s.picker
        physical = s.args.grasp_mode == "physical"
        pick = s.target.new_tensor(world_action[:2])
        place = s.target.new_tensor(world_action[2:])
        outcome = {"grasp_success": False, "dropped": False, "failure_reason": None}
        try:
            nodes = as_tensor(s.cloth.data.nodal_pos_w)[0]
            if physical:
                vertex = int(((nodes[:, :2] - pick) ** 2).sum(-1).argmin())
                pick_z = float(nodes[vertex, 2])
            else:
                slot = int(((s.corners()[:, :2] - pick) ** 2).sum(-1).argmin())
                vertex = int(s.corner_ids[slot])
                pick_z = float(nodes[vertex, 2])
            # The same world TCP goal convention is used in both modes.
            goal = s.target.new_tensor([float(pick[0]), float(pick[1]), min(.5, max(.18, pick_z + .12))])
            self.move(goal, 1.5)
            goal[2] = max(.004, pick_z + (.008 if physical else .015))
            self.move(goal, 1.5)
            s.action[0, 7] = -1
            self.hold(.5)
            if not picker.grasp(nodes[vertex].clone(), vertex_id=vertex):
                outcome["failure_reason"] = "pick_out_of_reach"
                return outcome
            before_z = float(as_tensor(s.cloth.data.nodal_pos_w)[0, vertex, 2])
            goal = s.tcp().clone()
            goal[2] += .04
            self.move(goal, .7)
            rise = float(as_tensor(s.cloth.data.nodal_pos_w)[0, vertex, 2]) - before_z
            outcome["lift_m"] = rise
            if rise < .015 or (physical and picker.drift() > .035):
                outcome["failure_reason"] = "lift_check_failed"
                return outcome
            outcome["grasp_success"] = True
            # End above the destination surface, then open, retreat, and settle.
            goal = s.tcp().clone()
            goal[:2] = place
            goal[2] = max(.008, float(s.corners()[:, 2].min()) + .006)
            self.move(goal, 2.5, arc_height=self.lift_height)
            if physical and picker.drift() > .05:
                outcome["dropped"] = True
                outcome["failure_reason"] = "slipped_during_transport"
        except RuntimeError as exc:
            if not str(exc).startswith("IK target not reached"):
                raise
            outcome["failure_reason"] = str(exc)
        finally:
            # Always open and detach, including on misses, slips, or exceptions.
            s.action[0, 7] = 1
            self.hold(.25)
            picker.release()
            goal = s.tcp().clone()
            goal[2] = min(.5, max(.18, float(goal[2]) + .08))
            try:
                self.move(goal, .8)
            except RuntimeError as exc:
                if not str(exc).startswith("IK target not reached"):
                    raise
                outcome["failure_reason"] = outcome["failure_reason"] or str(exc)
            finally:
                self.hold(s.args.settle_seconds)
        return outcome

    def close(self):
        self.session.close()
