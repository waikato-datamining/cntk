# Ubuntu 16.04, CUDA 9.0
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libpython3.5 && \
	apt-get install -y --no-install-recommends git-core openmpi-bin wget && \
	apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender-dev && \
	rm -rf /var/lib/apt/lists/* && \
	pip3 install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.7rc0.dev20190415-cp35-cp35m-linux_x86_64.whl && \
	pip3 install setuptools && pip3 install easydict pyyaml opencv-python Pillow

RUN cd /opt && git clone https://github.com/waikato-datamining/cntk.git
	
WORKDIR /opt/cntk/2.6/faster_rcnn/FasterRCNN