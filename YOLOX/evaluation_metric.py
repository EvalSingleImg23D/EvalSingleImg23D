#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# the Command to Run the Program:
# python3 SIGGRAPH_dir.py -n yolox-x -c ./yolox_x.pth --image_path assets/image.jpg --dir_path assets/rendered/ --img23D_model_name DG

# レンダリングした画像のディレクトリパスと元画像のパスを入力し, 上位5クラスの確信度とクラス名から評価を行う. 
# 推論を行うinferenceメソッドで面積が最大のbboxのみを返す. 

# vis_folder : image-to-3Dモデルごとに作成 (DG, TGS, LGM, etc)
# class_folder : クラスごと(元画像ごと)に作成 (airplane, bus, etc)
# class_folderの中に(original, 1, 2, ..., 120, csv, mp4)を格納

import argparse
import os, sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.append(parent_dir)

import cv2

import torch, torch.nn
import torch.nn.functional as F
import numpy as np
from transformers import ElectraTokenizer, ElectraModel

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis
from demo_utils import get_stem

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

embeddings = torch.load("ELECTRA_coco_embeddings.pt")
# embeddings[0]でperson(0番目)の埋め込みベクトルが取得できる. 

# コマンドライン引数の定義
def make_parser():
    parser = argparse.ArgumentParser("evaluation using YOLOX")
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="model name"
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, type=str, help="checkpoint for eval"
    )
    parser.add_argument(
        "--dir_path", help="path to the directory of rendering images of the generated object"
    )
    parser.add_argument(
        "--image_path", help="path to the original image"
    )
    parser.add_argument(
        "--img23D_model_name", type=str,
    )
    parser.add_argument(
        "-f", "--exp_file", default=None, type=str, help="please input your experiment description file",
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def softmax(x):
    x_cpu = [t.cpu().numpy() / 0.3 for t in x]    # xはテンソル
    exp_x = np.exp(x_cpu - np.max(x_cpu))   # オーバーフロー対策のために最大値を引く
    return exp_x / exp_x.sum()

def cos_sim(v1, v2):
    v1 = v1.flatten()  # 1次元に変換
    v2 = v2.flatten()  # 1次元に変換
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        decoder=None,
        device="gpu",
        fp16=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}                # img_info : id, file_name, height, width, raw_img, ratio
        if isinstance(img, str):
            img_info["file_name"] = get_stem(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio           # 画像サイズをモデルに合わせた縮尺係数

        # inference
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)       # 全部の情報を含んだテンソル ([1, 8400, 85])
            
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            
            # nms
            outputs, outputs_all = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            if outputs[0] is None:
                return None, None, img_info
            
            # 以降, 面積が最大のbboxのみを考える
            bbox = outputs[0][:,:4] / ratio # bboxの座標(が検出されたbbox分)
            largest_bbox = 0
            for box in range(len(bbox)):
                if((bbox[box][2]-bbox[box][0])*(bbox[box][3]-bbox[box][1]) > (bbox[largest_bbox][2]-bbox[largest_bbox][0])*(bbox[largest_bbox][3]-bbox[largest_bbox][1])):
                    largest_bbox = box
            outputs = outputs[0][largest_bbox]
            outputs_all = outputs_all[0][largest_bbox]
            return outputs, outputs_all, img_info

    # 推論結果を視覚化するメソッド
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]           # 画像サイズをモデルに合わせた縮尺係数
        img = img_info["raw_img"]           # 元画像データimg
        if output is None:
            return img
        output = output.cpu()

        bbox = output[0:4]                  # 推論結果outputからbboxの座標情報を取得
        bbox /= ratio                       # 元画像に対する座標に変換
        cls = output[6]                     # クラスのインデックス
        scores = output[4] * output[5]      # クラスごとの信頼度スコアを計算
        # BBOXの描画
        vis_res = vis(img, bbox, scores, cls, cls_conf, self.cls_names)
        return vis_res

# 元画像を物体検出し, 画像の保存とcsvへの書き込みを行う. 元画像の上位5位までの信頼度スコア・クラスのインデックスを返す. 
def original_image_process(predictor, class_folder, image_path):
    save_folder = os.path.join(
        class_folder, "input_image"
    )
    os.makedirs(save_folder, exist_ok=True)
    # 推論実行
    outputs, outputs_all, img_info = predictor.inference(image_path)
    if outputs[0] is None:
        print("cannot detect any objects.")
        sys.exit()

    result_image = predictor.visual(outputs, img_info, predictor.confthre)
    # crop_and_save_boxes(img_info, result_image, outputs[:4], outputs[4], outputs[6], predictor.cls_names, save_folder, predictor.confthre)
    # add_csv(img_info, outputs, "original_image", csv_path)
    
    # 全体の画像を保存
    save_file_name = os.path.join(save_folder, os.path.basename(image_path))
    cv2.imwrite(save_file_name, result_image)
    
    # 80クラスの確信度
    img_outputs_80 = outputs_all[7:].cpu().tolist()
    for i in range(80):
        img_outputs_80[i] *= outputs_all[4]
            
    # (値, インデックス) のペアを作成
    indexed_80 = list(enumerate(img_outputs_80))
    # 値に基づいて降順にソート
    indexed_80_sorted = sorted(indexed_80, key=lambda x: x[1], reverse=True)
    # ソートされたリストと対応するインデックスを抽出
    # scores_sorted = [x[1] for x in indexed_80_sorted]
    img_top5_index = [x[0] for x in indexed_80_sorted][:5]
    # for i in range(5):
        # print("i: ", COCO_CLASSES[img_top5_index[i]], "score: ", img_outputs_80[img_top5_index[i]])
    return img_outputs_80, img_top5_index


# 動画デモを実行
def imageflow_demo(predictor, class_folder, args):
    dir_path = args.dir_path
    image_path = args.image_path
    
    # 元画像の処理
    img_outputs_80, img_top5_index = original_image_process(predictor, class_folder, image_path)
    
    # フレームごと処理
    frame_count = 1
    r = []   
    files = get_image_list(dir_path)
    for image_name in files:
        frame_directory = os.path.join(
                class_folder, str(frame_count)
            )
        os.makedirs(frame_directory, exist_ok=True)
        
        # 推論実行
        outputs, outputs_all, frame_info = predictor.inference(image_name)
        # 何も検出されなかったフレームは全部ゼロにする
        if outputs is None:
            frame_count += 1
            r.append(0)
            # print(0)
            continue
        
        result_frame = predictor.visual(outputs, frame_info, predictor.confthre)
        
        # 各フレーム全体の検出結果を保存
        base_save_file_name = os.path.join(frame_directory, f"{frame_count}")
        save_file_name = f"{base_save_file_name}.jpg"
        cv2.imwrite(save_file_name, result_frame)
        
        # 指標値の計算
        # 80クラスの確信度
        outputs_80 = outputs_all[7:].cpu().tolist()
        for i in range(80):
            outputs_80[i] *= outputs_all[4]
        # (値, インデックス) のペアを作成
        indexed_80 = list(enumerate(outputs_80))
        # 値に基づいて降順にソート
        indexed_80_sorted = sorted(indexed_80, key=lambda x: x[1], reverse=True)
        # ソートされたリストと対応するインデックスを抽出
        frame_top5_index = [x[0] for x in indexed_80_sorted][:5]
        # for i in frame_top5_index:
            # print(COCO_CLASSES[i], outputs_80[i])
        
        # img_index_top5とframe_index_top5をマージ
        # セットを使って重複しない要素を取得
        C_GT_cup_i = list(set(img_top5_index) | set(frame_top5_index))
        # for i in range(len(C_GT_cup_i)):
            # print(COCO_CLASSES[C_GT_cup_i[i]])
        C_GT = img_top5_index
        # インデックスを使って要素を取得
        P_GT = [img_outputs_80[i] for i in C_GT_cup_i]
        # P_GT = [t.cpu().numpy() for t in P_GT]
        P_GT = softmax(P_GT)
        # print(P_GT)
        p_GT_norm = np.linalg.norm(P_GT)
        # print(p_GT_norm)
        P_i = [outputs_80[i] for i in C_GT_cup_i]
        # P_i = [t.cpu().numpy() for t in P_i]
        P_i = softmax(P_i)
        # print(P_i)
        
        p_i_norm = np.linalg.norm(P_i)
        # print(frame_count, COCO_CLASSES[frame_top5_index[0]], image_name)
                
        A = []
        for j in C_GT_cup_i:
            max = 0
            for k in C_GT:
                candidate = cos_sim(embeddings[j], embeddings[k])
                if candidate > max:
                    max = candidate
            A.append(max)
            # print(max)

        denominator = 0
        for j in range(len(C_GT_cup_i)):
            denominator += P_GT[j] * A[j] * P_i[j]
        radius = denominator / (p_GT_norm * p_i_norm)

        r.append(radius)
        # print(radius)
        frame_count += 1
    
    number_of_frame = frame_count - 1
    print("correct class: ", COCO_CLASSES[img_top5_index[0]])
    sum = 0
    for i in range(number_of_frame):
        sum += r[i]
    index_value = sum / number_of_frame
    print(index_value)
    return index_value

# エントリーポイント
def main(exp, args):
    # コマンドライン引数で変わらないところを普通の変数にした
    experiment_name = exp.exp_name
    exp.test_conf = 0.25
    exp.nmsthre = 0.45
    tsize = 640
    exp.test_size = (tsize, tsize)
    device = "gpu" if torch.cuda.is_available() else 'cpu'
    fp16 = False # Trueにすると半精度になる
    decoder = None

    # ./YOLOX_outpus/yolox_x/
    file_name = os.path.join(exp.output_dir, experiment_name)
    os.makedirs(file_name, exist_ok=True)

    # DG/, vis_res/, etc
    vis_folder = os.path.join(file_name, args.img23D_model_name)
    os.makedirs(vis_folder, exist_ok=True)
    
    # dog/, cat/, etc
    class_folder = os.path.join(vis_folder, get_stem(args.dir_path))
    os.makedirs(class_folder, exist_ok=True)

    model = exp.get_model() # モデルの取得

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()  # to FP16
    model.eval()
    
    
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cuda")
    # モデルの重みを読み込み
    model.load_state_dict(ckpt["model"])

    # 推論のためのpredictorオブジェクトを作成
    predictor = Predictor(
        model, exp, COCO_CLASSES, decoder,
        device, fp16,
    )
    # デモを行う関数を呼び出し
    # image_demo(predictor, vis_folder, args.image_path)
    imageflow_demo(predictor, class_folder, args)

# 直接実行する場合に実行される
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)