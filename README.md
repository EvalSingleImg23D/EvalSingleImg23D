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
