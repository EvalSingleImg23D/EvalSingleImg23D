import subprocess

# 名前と初期回転, カメラの距離のリストからなる辞書
dataset_name = {
    "airplane": [75, 0, 1.8],
    "bird": [120, 0, 2.0],
    "bottle": [90, -15, 1.8],
    "bowl": [90, 0, 1.8],
    "browncup": [90, 0, 1.8],
    "bus": [45, 0, 1.8],
    "chicken": [45, 0, 1.8],
    "clock": [90, 0, 1.8],
    "couch": [70, 0, 1.8],
    "cow": [30, 0, 1.8],
    "cup": [120, 0, 2.1],
    "dish": [90, 0, 1.8], 
    "hammer": [90, 0, 2.0],
    "hdd": [90, 0, 2.0],
    "horse": [165, 0, 1.8],
    "keyboard": [90, 0, 2.0],
    "laptop": [90, 0, 2.0],
    "lion": [150, 0, 2.0],
    "mouse": [45, 15, 2.1],
    "pitcher": [90, 0, 2.0],
    "rhino": [45, 0, 2.0],
    "scissors": [45, 0, 2.1],
    "shark": [45, 0, 2.0],
    "teapot": [120, 0, 2.0],
    "teddydog": [90, 0, 2.0],
    "thomas": [60, 0, 2.0],
    "towel": [90, 0, 1.6],
    "turtle": [75, 0, 2.1],
    "unicorn": [90, 0, 2.0],
    "wiiu": [15, 0, 2.0],
}

azi = 0
ele = -90 + 15
for name, rot_list in dataset_name.items():
    rot_azi = rot_list[0]
    rot_ele = rot_list[1]
    rad = rot_list[2]
    # 極以外
    while azi < 360: 
        while ele < 90:
            command = [
                "python3",
                "main_glb.py",
                "--name",
                "{}".format(name),
                "--azi",
                "{}".format(azi),
                "--ele",
                "{}".format(ele),
                "--rot_azi",
                "{}".format(rot_azi),
                "--rot_ele",
                "{}".format(rot_ele),
                "--radius",
                "{}".format(rad),
            ]
            subprocess.run(" ".join(command), shell=True)
            ele = ele + 15
        ele = -90 + 15
        azi = azi + 15
    azi = 0
    
    # 北極
    command = [
        "python3",
        "main_glb.py",
        "--name",
        "{}".format(name),
        "--azi",
        "0",
        "--ele",
        "90",
        "--rot_azi",
        "{}".format(rot_azi),
        "--rot_ele",
        "{}".format(rot_ele),
        "--radius",
        "{}".format(rad),
]
    subprocess.run(" ".join(command), shell=True)
    
    # 南極
    command = [
        "python3",
        "main_glb.py",
        "--name",
        "{}".format(name),
        "--azi",
        "0",
        "--ele",
        "-90",
        "--rot_azi",
        "{}".format(rot_azi),
        "--rot_ele",
        "{}".format(rot_ele),
        "--radius",
        "{}".format(rad),
    ]
    subprocess.run(" ".join(command), shell=True)
    