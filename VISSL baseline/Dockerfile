FROM python:3.8.18-bullseye

RUN pip install --upgrade pip
RUN pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu102_pyt181/download.html apex

WORKDIR /opt

RUN git clone --recursive https://github.com/facebookresearch/vissl.git

WORKDIR /opt/vissl


RUN git checkout v0.1.6 && \
	git checkout -b v0.1.6 && \
	pip install --progress-bar off -r requirements.txt && \
	pip install opencv-python && \
	pip uninstall -y classy_vision && \
	pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d && \
	pip install fairscale==0.4.6
	
RUN pip install -e ".[dev]"

RUN mkdir -p /tmp/configs/config && \
	wget -q -O configs/__init__.py https://dl.fbaipublicfiles.com/vissl/tutorials/configs/__init__.py && \
	wget -q -O configs/config/eval_resnet_8gpu_transfer_in1k_linear.yaml https://dl.fbaipublicfiles.com/vissl/tutorials/configs/quick_1gpu_resnet50_simclr.yaml && \
	wget -q  https://dl.fbaipublicfiles.com/vissl/tutorials/run_distributed_engines.py
	
ENTRYPOINT ["python"]
CMD ["run_distributed_engines.py","config=quick_1gpu_resnet50_simclr","config.DATA.TRAIN.DATA_SOURCES=[synthetic]"]
