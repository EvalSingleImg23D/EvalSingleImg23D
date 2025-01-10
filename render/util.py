import bpy
import math, mathutils

def normalize_obj(imported_objects, scale=1.0):
    for obj in imported_objects:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
        bbox = obj.bound_box
        max_dim = max([abs(bbox[0][i] - bbox[6][i]) for i in range(3)])
        obj_scale = scale / max_dim
        obj.scale = (obj_scale, obj_scale, obj_scale)
        obj.location = (0, 0, 0)


def set_camera_location(camera, azimuth, elevation, radius=2.0):
    elevation = 90 - elevation
    azimuth = azimuth + 180
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    
    x = radius * math.sin(elevation_rad) * math.cos(azimuth_rad)
    y = radius * math.sin(elevation_rad) * math.sin(azimuth_rad)
    z = radius * math.cos(elevation_rad)

    camera.location = (x, y, z)
    direction = camera.location - mathutils.Vector((0, 0, 0))
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    