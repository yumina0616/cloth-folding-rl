"""Single-node virtual grasp enforced inside PhysX, without nodal state writes.

Uses the installed OmniPhysicsVtxXformAttachment schema. Create only after Kit
has started. The anchor is an invisible Xform; update it BEFORE each physics step.
"""
from a_single_point_picker import VirtualPicker, as_tensor


def filter_gripper_contacts(env, env_ids=None):
    """Author virtual-gripper collision filters BEFORE physics views initialize."""
    from pxr import Usd, UsdPhysics

    stage = env.sim.stage
    root = env.scene.env_prim_paths[0]
    hand_names = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    robot_root = stage.GetPrimAtPath(root + "/Robot")
    robot_prims = list(Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies()))
    paths = [prim.GetPath() for prim in robot_prims
             if prim.HasAPI(UsdPhysics.CollisionAPI)
             and any(name in str(prim.GetPath()) for name in hand_names)]
    if not paths:
        raise RuntimeError("No hand/finger collision shapes found for virtual grasp filtering: "
                           + str([str(prim.GetPath()) for prim in robot_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]))
    cloth_root = stage.GetPrimAtPath(root + "/Deformable")
    colliders = [prim for prim in Usd.PrimRange(cloth_root)
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        raise RuntimeError("No cloth collision shapes found before physics startup")
    for prim in colliders:
        targets = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        for path in paths:
            targets.AddTarget(path)
    env.virtual_gripper_filtered_paths = [str(path) for path in paths]
    print(f"[virtual_grasp] Filtered {len(paths)} gripper collision shapes before physics startup", flush=True)


class AttachmentPicker(VirtualPicker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import omni.usd
        from pxr import UsdGeom

        self.backend = "physx_attachment"
        self._stage = omni.usd.get_context().get_stage()
        self._path = f"/World/VirtualGrasp_{id(self)}"
        self._frame = UsdGeom.Xform.Define(self._stage, self._path + "/Anchor")
        self._translate = self._frame.AddTranslateOp()
        self._attachment = None

    def update(self, dt=None):
        if not self.is_grasping:
            return
        from pxr import Gf, Sdf

        anchor = self._point(self.anchor_position())
        self._translate.Set(Gf.Vec3d(*anchor.tolist()))
        if self._attachment is None:
            meshes = self.cloth.root_view.simulation_mesh_prim_paths
            mesh_path = meshes[self.env_id]
            if isinstance(mesh_path, (list, tuple)):
                if len(mesh_path) != 1:
                    raise RuntimeError(f"Expected one simulation mesh: {mesh_path}")
                mesh_path = mesh_path[0]
            attachment = self._stage.DefinePrim(self._path + "/Constraint", "OmniPhysicsVtxXformAttachment")
            attachment.GetRelationship("omniphysics:src0").SetTargets([Sdf.Path(mesh_path)])
            attachment.GetRelationship("omniphysics:src1").SetTargets([self._frame.GetPath()])
            attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set([self.vertex_id])
            attachment.GetAttribute("omniphysics:localPositionsSrc1").Set([Gf.Vec3f(*self._offset.tolist())])
            attachment.GetAttribute("omniphysics:attachmentEnabled").Set(True)
            self._attachment = attachment
        self._previous_target = (anchor + self._offset).clone()

    def release(self):
        if self._attachment is not None:
            self._attachment.GetAttribute("omniphysics:attachmentEnabled").Set(False)
            self._stage.RemovePrim(self._attachment.GetPath())
            self._attachment = None
        self.vertex_id = None
        self._offset = self._previous_target = self._saved_target = None


def create_picker(cloth, anchor_position, *, grasp_mode="virtual", **kwargs):
    if grasp_mode != "virtual":
        raise ValueError("Only virtual grasp is supported")
    return AttachmentPicker(cloth, anchor_position, **kwargs)
