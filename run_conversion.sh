#!/bin/bash
inputh5="../r0-2022-01-25-ITJohnFrontV2-018.h5"
nframes=20 # 300
nframes_to_draw=10 # 1
mask_folder="masks/deeplabraw"

python3 stage_h5_to_pcd.py -h5 $inputh5 \
	-n $nframes \
	-nd $nframes_to_draw \
	-m $mask_folder
