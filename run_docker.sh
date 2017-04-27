#!/usr/bin/env bash

nvidia-docker run -it \
	-v /directory/containing/all/ten/image/class/directories:/root/images/release_4_1_17/release_v2/aligned/2D \
	-v $(pwd):/root/torch_integrated_cell \
	floydhub/dl-docker:gpu \
	bash
