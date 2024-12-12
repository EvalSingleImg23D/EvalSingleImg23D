# EvalSingleImg23D

## Install

(CUDA12.2, python3.10.12)

- Environment setup

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install wheel
cd YOLOX/
pip3 install -r requirement.txt
pip3 install -v -e .
pip3 install bpy mathutils transformers
```

- Download pretrained weight

```
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
```

## Evaluation demo

- Render 3D object from various viewpoints
  - coming soom
- Calculate evaluation metric

```
cd YOLOX/
python3 evaluation_metric.py -n yolox-x -c ./yolox_x.pth --image_path assets/successful_example/horse.png --dir_path assets/successful_example/horse/ --img23D_model_name [model_name]
python3 evaluation_metric.py -n yolox-x -c ./yolox_x.pth --image_path assets/unsuccessful_example/airplane.png --dir_path assets/unsuccessful_example/airplane/ --img23D_model_name [model_name]
```

## Memo

- Files changed from the original YOLOX

  - yolox/utils/visualize.py
  - yolox/utils/boxes.py

- TO-DO
  - release rendering code
  - Package this repository so that it can be installed with pip install ......

## Acknowledgement

In this study, we utilized the YOLOX code. We sincerely thank the developers for sharing the code.

https://github.com/Megvii-BaseDetection/YOLOX

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

## Citation

```latex
@inproceedings{EvalSingleImg23D,
author = {Uchida, Yuiko and Togo, Ren and Maeda, Keisuke and Ogawa, Takahiro and Haseyama, Miki},
title = {An Evaluation Metric for Single Image-to-3D Models Based on Object Detection Perspective},
year = {2024},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3681758.3697992},
doi = {10.1145/3681758.3697992},
booktitle = {SIGGRAPH Asia 2024 Technical Communications},
articleno = {31},
numpages = {4},
location = {
},
series = {SA '24}
}
```
