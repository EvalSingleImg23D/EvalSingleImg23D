import argparse

import bpy
import mathutils


def make_parser():
    parser = argparse.ArgumentParser("normalize obj into [-0.5, 0.5] cube")
    parser.add_argument(
        "--obj_path", type=str, default=None, help="path to input obj"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="path to output obj"
    )
    return parser

def normalize_obj(obj_path, output_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]

    min_x, min_y, min_z = [min([v.co[i] for v in obj.data.vertices]) for i in range(3)]
    max_x, max_y, max_z = [max([v.co[i] for v in obj.data.vertices]) for i in range(3)]

    # calc center
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    max_dimension = max(max_x - min_x, max_y - min_y, max_z - min_z)
    scale_factor = 1.0 / max_dimension

    # translation and scaling 
    bpy.ops.object.mode_set(mode='OBJECT')
    for v in obj.data.vertices:
        v.co = (v.co - mathutils.Vector((center_x, center_y, center_z))) * scale_factor

    bpy.ops.object.transform_apply(location=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # export normalized mesh
    bpy.ops.wm.obj_export(filepath=output_path)

if __name__ == "__main__":
    args = make_parser().parse_args()
    normalize_obj(args.obj_path, args.output_path)