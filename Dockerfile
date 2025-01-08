FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install wheel \
                opencv-python==4.5.5.64

# WORKDIR /workspace/YOLOX
# COPY YOLOX/ /workspace/YOLOX

# RUN ls -la /workspace/YOLOX
COPY YOLOX/requirements.txt /workspace/YOLOX/requirements.txt
COPY YOLOX/ /workspace/YOLOX/

RUN python3.10 -m pip install -r /workspace/YOLOX/requirements.txt \
    && python3.10 -m pip install -v -e /workspace/YOLOX/

RUN pip3 install bpy mathutils transformers
