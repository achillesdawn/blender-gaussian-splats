import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import Mesh, FloatVectorAttribute, ByteColorAttribute, NodesModifier

from .plyfile import PlyData
from pathlib import Path
from typing import cast

import numpy as np

import os
import math


def RS_matrix(quat, scale):
    matrix = []

    length = 1 / math.sqrt(
        quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]
    )

    x = quat[0] * length
    y = quat[1] * length
    z = quat[2] * length
    w = quat[3] * length

    matrix.append(scale[0] * (1 - 2 * (z * z + w * w)))
    matrix.append(scale[0] * (2 * (y * z + x * w)))
    matrix.append(scale[0] * (2 * (y * w - x * z)))
    matrix.append(scale[1] * (2 * (y * z - x * w)))
    matrix.append(scale[1] * (1 - 2 * (y * y + w * w)))
    matrix.append(scale[1] * (2 * (z * w + x * y)))
    matrix.append(scale[2] * (2 * (y * w + x * z)))
    matrix.append(scale[2] * (2 * (z * w - x * y)))
    matrix.append(scale[2] * (1 - 2 * (y * y + z * z)))

    return matrix


class SNA_OT_IMPORT_PLY(bpy.types.Operator, ImportHelper):
    bl_idname = "gsplat.import_ply"
    bl_label = "Import PLY As Splats"
    bl_description = "Import a .PLY"

    bl_options = {"REGISTER", "UNDO"}

    filter_glob: bpy.props.StringProperty(default="*.ply", options={"HIDDEN"})  # type: ignore

    @classmethod
    def poll(cls, context):
        enabled_modes = ["OBJECT"]
        return context.mode in enabled_modes and bpy.app.version >= (4, 2, 0)

    @staticmethod
    def append_nodes(object_name: str):
        blender_path = "assets.blend"
        path = Path(__file__).resolve().parent / blender_path
        inner_path = "NodeTree"
        filepath = path / inner_path / object_name

        print(f"importing from {filepath.as_posix()}...")

        bpy.ops.wm.append(
            filepath=filepath.as_posix(),
            directory=(path / inner_path).as_posix(),
            filename=object_name,
        )

    @staticmethod
    def append_materials(object_name: str):
        blender_path = "assets.blend"
        path = Path(__file__).resolve().parent / blender_path
        inner_path = "Material"
        filepath = path / inner_path / object_name

        print(f"importing from {filepath.as_posix()}...")

        bpy.ops.wm.append(
            filepath=filepath.as_posix(),
            directory=(path / inner_path).as_posix(),
            filename=object_name,
        )

    def execute(self, context):
        # append nodes
        self.append_nodes("render")
        self.append_nodes("set_material")
        self.append_nodes("sort")
        self.append_materials("gaussian_splat_material")

        ply_import_path = self.filepath

        plydata = PlyData.read(ply_import_path)

        file_base_name = os.path.splitext(os.path.basename(ply_import_path))[0]

        object_name = f"{file_base_name}"

        center = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        splat_count = int(len(center))

        N = splat_count

        if "opacity" in plydata.elements[0]:
            log_opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            opacities = 1 / (1 + np.exp(-log_opacities))
        else:
            log_opacities = np.asarray(1)[..., np.newaxis]
            opacities = 1 / (1 + np.exp(-log_opacities))

        opacities = opacities.flatten()

        # features
        features_dc = np.zeros((N, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # scales and rotations
        log_scales = np.stack(
            (
                np.asarray(plydata.elements[0]["scale_0"]),
                np.asarray(plydata.elements[0]["scale_1"]),
                np.asarray(plydata.elements[0]["scale_2"]),
            ),
            axis=1,
        )
        scales = np.exp(log_scales)
        quats = np.stack(
            (
                np.asarray(plydata.elements[0]["rot_0"]),
                np.asarray(plydata.elements[0]["rot_1"]),
                np.asarray(plydata.elements[0]["rot_2"]),
                np.asarray(plydata.elements[0]["rot_3"]),
            ),
            axis=1,
        )

        vertices = []
        indices = []
        for i in range(splat_count):
            vertices.append((-2.0, -2.0, float(i)))
            vertices.append((2.0, -2.0, float(i)))
            vertices.append((2.0, 2.0, float(i)))
            vertices.append((-2.0, 2.0, float(i)))
            b = i * 4
            indices.append((b, b + 1, b + 2))
            indices.append((b, b + 2, b + 3))

        mesh: Mesh = bpy.data.meshes.new(name=object_name)

        mesh.from_pydata(vertices, [], indices)

        ob: bpy.types.Object = bpy.data.objects.new(object_name, mesh)

        Vrk_1 = [0.0] * splat_count * 2 * 1
        Vrk_2 = [0.0] * splat_count * 2 * 1
        Vrk_3 = [0.0] * splat_count * 2 * 1
        Vrk_4 = [0.0] * splat_count * 2 * 1
        Vrk_5 = [0.0] * splat_count * 2 * 1
        Vrk_6 = [0.0] * splat_count * 2 * 1

        center_data = [0.0] * splat_count * 2 * 3
        color = [0.0] * splat_count * 2 * 4

        SH_0 = 0.28209479177387814
        for i in range(splat_count):
            RS = RS_matrix(quats[i], scales[i])
            # Covariance Matrix
            vrk_1 = RS[0] * RS[0] + RS[3] * RS[3] + RS[6] * RS[6]
            vrk_2 = RS[0] * RS[1] + RS[3] * RS[4] + RS[6] * RS[7]
            vrk_3 = RS[0] * RS[2] + RS[3] * RS[5] + RS[6] * RS[8]
            vrk_4 = RS[1] * RS[1] + RS[4] * RS[4] + RS[7] * RS[7]
            vrk_5 = RS[1] * RS[2] + RS[4] * RS[5] + RS[7] * RS[8]
            vrk_6 = RS[2] * RS[2] + RS[5] * RS[5] + RS[8] * RS[8]
            Vrk_1[2 * i + 0] = vrk_1
            Vrk_1[2 * i + 1] = vrk_1
            Vrk_2[2 * i + 0] = vrk_2
            Vrk_2[2 * i + 1] = vrk_2
            Vrk_3[2 * i + 0] = vrk_3
            Vrk_3[2 * i + 1] = vrk_3
            Vrk_4[2 * i + 0] = vrk_4
            Vrk_4[2 * i + 1] = vrk_4
            Vrk_5[2 * i + 0] = vrk_5
            Vrk_5[2 * i + 1] = vrk_5
            Vrk_6[2 * i + 0] = vrk_6
            Vrk_6[2 * i + 1] = vrk_6
            # To this:
            center_data[6 * i + 0] = center[i][0]
            center_data[6 * i + 1] = center[i][1]
            center_data[6 * i + 2] = center[i][2]
            center_data[6 * i + 3] = center[i][0]
            center_data[6 * i + 4] = center[i][1]
            center_data[6 * i + 5] = center[i][2]
            # Colors
            R = features_dc[i][0][0] * SH_0 + 0.5
            G = features_dc[i][1][0] * SH_0 + 0.5
            B = features_dc[i][2][0] * SH_0 + 0.5
            A = opacities[i]
            color[8 * i + 0] = R
            color[8 * i + 1] = G
            color[8 * i + 2] = B
            color[8 * i + 3] = A
            color[8 * i + 4] = R
            color[8 * i + 5] = G
            color[8 * i + 6] = B
            color[8 * i + 7] = A

        center_attr = cast(
            FloatVectorAttribute,
            mesh.attributes.new(name="center", type="FLOAT_VECTOR", domain="FACE"),
        )

        center_attr.data.foreach_set("vector", center_data)

        color_attr = cast(
            ByteColorAttribute,
            mesh.attributes.new(name="color", type="FLOAT_COLOR", domain="FACE"),
        )

        color_attr.data.foreach_set("color", color)

        for idx, data in enumerate([Vrk_1, Vrk_2, Vrk_3, Vrk_4, Vrk_5, Vrk_6]):
            Vrk_attr = mesh.attributes.new(
                name=f"Vrk_{idx + 1}", type="FLOAT", domain="FACE"
            )
            Vrk_attr.data.foreach_set("value", data)

        
        render_nodes = bpy.data.node_groups["render"]
        sort_nodes = bpy.data.node_groups["sort"]

        render_modifier = cast(
            NodesModifier, ob.modifiers.new(name="gaussian_render", type="NODES")
        )

        render_modifier.node_group = render_nodes

        sort_modifier = cast(
            NodesModifier, ob.modifiers.new(name="gaussian_sort", type="NODES")
        )

        sort_modifier.node_group = sort_nodes


        material = bpy.data.materials.get("gaussian_splat_material")

        assert material
        
        material.surface_render_method = "BLENDED"
        mesh.materials.append(material)
        
        assert context.collection
        assert context.view_layer

        context.collection.objects.link(ob)

        context.view_layer.objects.active = ob

        ob.select_set(True)

        print(f"Created Gaussian Splat object {ob.name}")

        # if bpy.context.scene.sna_kiri3dgs_import_face_alignment == "To X Axis":
        #     sna_align_active_values_to_x_function_execute_03E8D()
        # else:
        #     if bpy.context.scene.sna_kiri3dgs_import_auto_rotate:
        #         if bpy.context.scene.sna_kiri3dgs_import_face_alignment == "To Y Axis":
        #             sna_align_active_values_to_z_function_execute_62C4D()
        #         else:
        #             sna_align_active_values_to_y_function_execute_89335()
        #     else:
        #         if bpy.context.scene.sna_kiri3dgs_import_face_alignment == "To Y Axis":
        #             sna_align_active_values_to_y_function_execute_89335()
        #         else:
        #             sna_align_active_values_to_z_function_execute_62C4D()

        # geonodemodreturn_0_bf551 = sna_add_geo_nodes__append_group_2D522_BF551(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Store_Origpos_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Store_Origpos_GN",
        # )
        # bpy.ops.object.modifier_move_to_index(
        #     "INVOKE_DEFAULT", modifier="KIRI_3DGS_Store_Origpos_GN", index=0
        # )
        # bpy.ops.object.modifier_apply(
        #     "INVOKE_DEFAULT", modifier="KIRI_3DGS_Store_Origpos_GN"
        # )
        # bpy.ops.object.modifier_apply("INVOKE_DEFAULT", modifier="KIRI_3DGS_Render_GN")
        # geonodemodreturn_0_91587 = sna_add_geo_nodes__append_group_2D522_91587(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Remove_Stray_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Remove_Stray_GN",
        # )
        # geonodemodreturn_0_8e257 = sna_add_geo_nodes__append_group_2D522_8E257(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Camera_Cull_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Camera_Cull_GN",
        # )
        # geonodemodreturn_0_dde79 = sna_add_geo_nodes__append_group_2D522_DDE79(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Crop_Box_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Crop_Box_GN",
        # )
        # geonodemodreturn_0_eb4fd = sna_add_geo_nodes__append_group_2D522_EB4FD(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Colour_Edit_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Colour_Edit_GN",
        # )
        # geonodemodreturn_0_592e9 = sna_add_geo_nodes__append_group_2D522_592E9(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Decimate_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Decimate_GN",
        # )
        # geonodemodreturn_0_b6203 = sna_add_geo_nodes__append_group_2D522_B6203(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Render_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Render_GN",
        # )
        # bpy.context.view_layer.objects.active.modifiers["KIRI_3DGS_Render_GN"][
        #     "Socket_55"
        # ] = True
        # geonodemodreturn_0_5fbae = sna_add_geo_nodes__append_group_2D522_5FBAE(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Animate_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Animate_GN",
        # )
        # geonodemodreturn_0_74b9d = sna_add_geo_nodes__append_group_2D522_74B9D(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Set_Material_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Set_Material_GN",
        # )
        # geonodemodreturn_0_03222 = sna_add_geo_nodes__append_group_2D522_03222(
        #     os.path.join(
        #         os.path.dirname(__file__), "assets", "3DGS Render APPEND.blend"
        #     ),
        #     "KIRI_3DGS_Sorter_GN",
        #     bpy.context.view_layer.objects.active,
        #     "KIRI_3DGS_Sorter_GN",
        # )
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Camera_Cull_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Crop_Box_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Colour_Edit_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Remove_Stray_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Decimate_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Render_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Animate_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Sorter_GN"
        # ].show_viewport = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Camera_Cull_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Crop_Box_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Colour_Edit_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Remove_Stray_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Decimate_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Animate_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Sorter_GN"
        # ].show_render = False
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_active_object_update_mode = (
        #     "Disable Camera Updates"
        # )
        # bpy.context.view_layer.objects.active.modifiers["KIRI_3DGS_Render_GN"][
        #     "Socket_50"
        # ] = 1
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Render_GN"
        # ].show_on_cage = True
        # bpy.context.view_layer.objects.active.modifiers[
        #     "KIRI_3DGS_Render_GN"
        # ].show_in_editmode = True
        # bpy.context.view_layer.objects.active.modifiers["KIRI_3DGS_Set_Material_GN"][
        #     "Socket_2"
        # ] = bpy.data.materials["KIRI_3DGS_Render_Material"]
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_active_object_enable_active_camera = False
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_active_object_update_mode = (
        #     "Disable Camera Updates"
        # )
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_modifier_enable_animate = (
        #     False
        # )
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_modifier_enable_camera_cull = False
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_modifier_enable_colour_edit = False
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_modifier_enable_crop_box = (
        #     False
        # )
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_modifier_enable_decimate = (
        #     False
        # )
        # bpy.context.view_layer.objects.active.sna_kiri3dgs_modifier_enable_remove_stray = False
        # bpy.context.scene.sna_kiri3dgs_lq_mode__hq_mode = "LQ Mode (Dithered Alpha)"
        # bpy.data.materials[
        #     "KIRI_3DGS_Render_Material"
        # ].surface_render_method = "DITHERED"
        # bpy.context.view_layer.objects.active.update_tag(
        #     refresh={"OBJECT"},
        # )
        # if bpy.context and bpy.context.screen:
        #     for a in bpy.context.screen.areas:
        #         a.tag_redraw()
        # if bpy.context.scene.sna_kiri3dgs_import_auto_rotate:
        #     bpy.context.view_layer.objects.active.rotation_euler = (
        #         math.radians(-90.0),
        #         0.0,
        #         math.radians(180.0),
        #     )
        return {"FINISHED"}
