"""A closed, thick render mesh following the simulated cloth, with terry detail.

The extra shell is graphics only: it never writes simulated positions/velocities
and does not add duplicate physical bodies or attachment constraints.
"""
from pathlib import Path

import numpy as np
import torch

from a_single_point_picker import as_tensor


def make_terry_textures(directory):
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    color_path, normal_path = directory / "terry_color.png", directory / "terry_normal.png"
    if color_path.exists() and normal_path.exists():
        return color_path, normal_path
    n = 1024
    y, x = np.mgrid[:n, :n].astype(np.float32) / n
    # Periodic small loops with finer crossing yarns, rather than large mesh bumps.
    loops = np.sin(2 * np.pi * 32 * x + .5 * np.sin(2 * np.pi * 32 * y))
    loops *= np.sin(2 * np.pi * 32 * y)
    yarn = .12 * np.sin(2 * np.pi * 160 * x) + .12 * np.sin(2 * np.pi * 160 * y)
    height = loops + yarn
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 1.2
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 1.2
    normals = np.stack((-dx, -dy, np.ones_like(dx)), -1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    Image.fromarray(np.uint8(np.clip(normals * .5 + .5, 0, 1) * 255)).save(normal_path)
    grain = np.random.default_rng(0).normal(0, .012, x.shape)
    color = np.array([.88, .79, .49]) * (1 + .035 * loops[..., None] + grain[..., None])
    Image.fromarray(np.uint8(np.clip(color, 0, 1) * 255)).save(color_path)
    return color_path, normal_path


class TowelVisual:
    def __init__(self, cloth, thickness, texture_dir):
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self.cloth, self.thickness = cloth, thickness
        self._Vt, self._Gf = Vt, Gf
        stage = omni.usd.get_context().get_stage()
        mesh_path = cloth.root_view.simulation_mesh_prim_paths[0]
        if isinstance(mesh_path, (list, tuple)):
            mesh_path = mesh_path[0]
        source = UsdGeom.Mesh.Get(stage, mesh_path)
        counts = np.asarray(source.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise RuntimeError("Towel visual expects a triangle simulation mesh")
        triangles = np.asarray(source.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        nodes = as_tensor(cloth.data.nodal_pos_w)[0]
        self.triangles = torch.as_tensor(triangles.copy(), device=nodes.device)
        self.vertex_count = len(nodes)
        if int(triangles.max()) >= self.vertex_count:
            raise RuntimeError("Simulation and rendering node indices do not match")
        # Single-use directed edges form the sheet's boundary. Close the shell
        # with side-wall triangles; do not simulate these extra render vertices.
        directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        _, inverse, occurrences = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        boundary = directed[occurrences[inverse] == 1]
        k = self.vertex_count
        sides = []
        for a, b in boundary:
            sides.extend([[b, a, a + k], [b, a + k, b + k]])
        faces = np.concatenate([triangles, triangles[:, ::-1] + k, np.asarray(sides)], axis=0)
        self.mesh = UsdGeom.Mesh.Define(stage, "/World/TowelRender")
        self.mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        self.mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.SetNormalsInterpolation("vertex")
        points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float32)
        uv = (points[:, :2] - points[:, :2].min(0)) / np.maximum(np.ptp(points[:, :2], axis=0), 1e-8)
        st = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
        st.Set(Vt.Vec2fArray.FromNumpy(np.concatenate([uv, uv]).astype(np.float32)))

        color_path, normal_path = make_terry_textures(texture_dir)
        material = UsdShade.Material.Define(stage, "/World/TowelMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/TowelMaterial/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.96)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        reader = UsdShade.Shader.Define(stage, "/World/TowelMaterial/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        transform = UsdShade.Shader.Define(stage, "/World/TowelMaterial/Tile")
        transform.CreateIdAttr("UsdTransform2d")
        transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(6., 6.))
        for name, path, channel in [("Color", color_path, "diffuseColor"), ("Normal", normal_path, "normal")]:
            texture = UsdShade.Shader.Define(stage, f"/World/TowelMaterial/{name}")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(path)))
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            if channel == "normal":
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2., 2., 2., 1.))
                texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1., -1., -1., 0.))
            else:
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            shader.CreateInput(channel, Sdf.ValueTypeNames.Normal3f if channel == "normal" else Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
        # Hide the zero-thickness render surface while retaining its physics.
        UsdGeom.Imageable(source.GetPrim()).MakeInvisible()
        self.update()

    def update(self):
        nodes = as_tensor(self.cloth.data.nodal_pos_w)[0]
        face = nodes[self.triangles]
        weighted = torch.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0], dim=-1)
        normals = torch.zeros_like(nodes)
        for i in range(3):
            normals.index_add_(0, self.triangles[:, i], weighted)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        shell = torch.cat((nodes + normals * self.thickness / 2, nodes - normals * self.thickness / 2))
        shell_np = shell.detach().cpu().numpy().astype(np.float32)
        self.mesh.GetPointsAttr().Set(self._Vt.Vec3fArray.FromNumpy(shell_np))
        normal_np = torch.cat((normals, -normals)).detach().cpu().numpy().astype(np.float32)
        self.mesh.GetNormalsAttr().Set(self._Vt.Vec3fArray.FromNumpy(normal_np))
        self.mesh.GetExtentAttr().Set([self._Gf.Vec3f(*shell_np.min(0).tolist()),
                                      self._Gf.Vec3f(*shell_np.max(0).tolist())])
