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
