import bpy
import math
import argparse
from util import set_camera_location
import os

def make_parser():
    parser = argparse.ArgumentParser("render obj from any direction")
    parser.add_argument("--name", type=str, default="horse")
    parser.add_argument("--azi", type=float, default=0)
    parser.add_argument("--ele", type=float, default=0)
    parser.add_argument("--rot_azi", type=float, default=0)
    parser.add_argument("--rot_ele", type=float, default=0)
    parser.add_argument("--radius", type=float, default=2.0)
    return parser

args = make_parser().parse_args()
name = args.name
radius = args.radius

glb_file_path = f"./assets/generated_object/{name}.glb"
output_image_path = f"./assets/{name}/"
print("GLB file path:", glb_file_path)
print("File exists:", os.path.exists(glb_file_path))

# Delete existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import GLB file
bpy.ops.import_scene.gltf(filepath=glb_file_path)
imported_objects = bpy.context.selected_objects

# for obj in imported_objects:
    # obj.scale.z = -1
# Set camera and light
bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
bpy.context.scene.camera = bpy.context.object
camera = bpy.context.object

bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'STUDIO'
bpy.context.scene.display.shading.studio_light = 'paint.sl'
bpy.context.scene.display.shading.color_type = 'TEXTURE'

# Rendering settings
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True

azimuth = args.azi
elevation = args.ele
rot_azi = args.rot_azi
rot_ele = args.rot_ele
set_camera_location(camera, azimuth + rot_azi, elevation + rot_ele, radius)
scene.render.filepath = output_image_path + f"{name}_{azimuth}_{elevation}.png"

bpy.ops.render.render(write_still=True)

# Update 3D view
bpy.context.view_layer.update()
