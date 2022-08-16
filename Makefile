ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

install:
	python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html;
	pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html;
	pip install -r ./Mask2Former/requirements.txt;
	pip install tensorboard

build:
	export FORCE_CUDA=1
	export CUDA_HOME=/usr/local/cuda-11.1/;
	export Mask2Former=$(ROOT_DIR)/Mask2Former;
	export CUDA_LAUNCH_BLOCKING=1
	export TORCH_CUDA_ARCH_LIST='7.5'
	python ./Mask2Former/mask2former/modeling/pixel_decoder/ops/setup.py build install;

uninstall:
	rm -rf Mask2Former dist build
	pip uninstall detectron2 -y
	pip uninstall torch torchvision
