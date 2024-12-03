#!/usr/bin/env python3

import os
import cv2

# 検出された物体のbboxを切り出して保存
def crop_and_save_boxes(img_info, image, box, score, cls_id, class_names, save_folder, confthre):
    ratio = img_info["ratio"]           # 画像サイズをモデルに合わせた縮尺係数
    box /= ratio                        # 元画像に対する座標に変換(重要!!!)
    cls_id = int(cls_id)                # クラスID

    if score < confthre:
        return

    x0, y0, x1, y1 = map(int, box)

    # クロップ
    cropped_box = image[y0:y1, x0:x1]

    file_name = os.path.join(save_folder, f"{cls_id}")
    save_file_name = f"{file_name}.jpg"
    if cropped_box.size > 0:
        cv2.imwrite(save_file_name, cropped_box)