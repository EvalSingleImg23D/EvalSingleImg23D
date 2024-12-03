#!/usr/bin/env python3

import os
# import pandas as pd
from yolox.data.datasets import COCO_CLASSES

# 拡張子とアンダーバー以降を除去し, パスの語幹のみ抽出する関数
# image_rgba.jpg -> image
def get_stem(path):
    no_extension_path  = os.path.splitext(os.path.basename(path))[0]
    prefix = no_extension_path.split('_')
    return prefix[0]

# class_folderにvideo名.csvを作成し, ヘッダを書き込んでパスを返す
# def make_csv(class_folder, video_path):
#     no_extension_path  = os.path.splitext(os.path.basename(video_path))[0]
#     prefix = no_extension_path.split('_')
#     csv_path = os.path.join(class_folder, f'{prefix[0]}.csv')

#     df = pd.DataFrame(
#         columns=[
#                 'frame', 
#                 'x1',
#                 'y1',
#                 'x2',
#                 'y2',
#                 '1st_conf',
#                 '1st_class_idx',
#                 '1st_class_name'])
#     df.to_csv(csv_path, index=False)
#     return csv_path

# # csvに保存
# def add_csv(img_info, outputs, frame_count, csv_path):
#     ratio = img_info["ratio"]
#     final_box = outputs[:4]/ratio                             # bboxの座標
#     final_score = outputs[4] * outputs[5]                     # 1位クラスの信頼度スコア
#     final_cls_index = outputs[6]                              # 1位クラスのインデックス
    
#     # データフレームの作成
#     result = {
#             'frame' : frame_count,
#             'x1': final_box[0].tolist(),
#             'y1': final_box[1].tolist(),
#             'x2': final_box[2].tolist(),
#             'y2': final_box[3].tolist(),
#             '1st_conf': final_score.tolist(),
#             '1st_class_idx': final_cls_index.tolist(),
#             '1st_class_name': [COCO_CLASSES[int(final_cls_index)]],
#     }
#     df = pd.DataFrame(result)
#     df.to_csv(csv_path, mode='a', header=False, index=False)

# # csvに空行を保存
# def add_blank_csv(frame_count, csv_path):
#     result = {
#             'frame': frame_count, 'x1': [None], 'y1': [None], 'x2': [None], 'y2': [None], '1st_conf': [0], '1st_class_idx': [None], '1st_class_name': [None],}
#     df = pd.DataFrame(result)
#     df.to_csv(csv_path, mode='a', header=False, index=False)