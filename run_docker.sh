#!/usr/bin/env bash

nvidia-docker run -it \
	-v /allen/aics/modeling/data/release_4_1_17/results_v2/aligned/2D:/root/images/release_4_1_17/release_v2/aligned/2D \
	-v /allen/aics/modeling/rorydm/torch_integrated_cell:/root/torch_integrated_cell \
	floydhub/dl-docker:gpu \
	bash
