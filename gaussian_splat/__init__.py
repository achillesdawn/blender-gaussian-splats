import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    BoolProperty,
    FloatProperty,
    EnumProperty,
    PointerProperty,
)

from pathlib import Path
from .splats import SNA_OT_IMPORT_PLY

bl_info = {
    "name": "GSPLAT",
    "author": "https://ko-fi.com/olivio",
    "version": (1, 0, 0),
    "blender": (4, 3, 2),
    "description": "View Gaussian Splats in blender",
    "category": "Tools",
}


class Config:
    normal_name = bl_info["name"]
    caps_name = "GSPLAT"
    lower_case_name = caps_name.lower()
    addon_global_var_name = "gsplat"

    panel_prefix = caps_name + "_PT_"
    operator_prefix = caps_name + "_OT_"


class GSPLAT_PROPERTIES(bpy.types.PropertyGroup):
    pass


class GSPLAT_PT_VIEWPORT_SIDE_PANEL(bpy.types.Panel):
    bl_idname = Config.panel_prefix + "viewport"
    bl_label = Config.normal_name
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "GSPLAT"

    @staticmethod
    def get_props():
        return getattr(bpy.context.scene, Config.addon_global_var_name)

    def draw(self, context):
        layout = self.layout
        props = self.get_props()

        col = layout.column()

        col.scale_y = 1.6
        col.operator(SNA_OT_IMPORT_PLY.bl_idname, text="import .ply")


classes = [GSPLAT_PROPERTIES, GSPLAT_PT_VIEWPORT_SIDE_PANEL, SNA_OT_IMPORT_PLY]


def set_properties():
    setattr(
        bpy.types.Scene,
        Config.addon_global_var_name,
        bpy.props.PointerProperty(type=GSPLAT_PROPERTIES),
    )


def del_properties():
    delattr(bpy.types.Scene, Config.addon_global_var_name)


def register():
    from bpy.utils import register_class

    for cls in classes:
        register_class(cls)

    set_properties()


def unregister():
    from bpy.utils import unregister_class

    for cls in reversed(classes):
        unregister_class(cls)

    del_properties()


if __name__ == "__main__":
    register()
