# Update of the Dockerfile at https://towardsdatascience.com/detectron2-the-basic-end-to-end-tutorial-5ac90e2f90e3
FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get install -y python3-opencv

# Detectron2 prerequisites
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python