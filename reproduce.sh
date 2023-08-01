

VERSION=2022.2.0

docker run --rm -it --user 0 \
	-e PYTHONEXE=python3 \
	-v `pwd`:/src openvino/ubuntu18_dev:$VERSION \
	/bin/bash -c /src/vino/convertToVino.sh
