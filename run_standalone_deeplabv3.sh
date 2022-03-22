#!/bin/bash

#
# need to update the parameters: dockertag, inputdir, outputdir
#

# for running with raw deeplabv3 model, use: inference_rawmodel.py
# with re-trained deeplabv3 model, use: inference.py
# details can be found in:
#   gepetto/third_party/BgSegmentation/BgSegmentation_DeepLabRetrain
inference_script="inference_rawmodel.py"

dockertag="3af2dc_jie"
dockerimage=bgsegmentation_deeplab_retrain:${dockertag}
# UPDATE!!
inputdir=/path/dir/to/input/images/
outputdir=/path/dir/to/output/masks/

mkdir -p $outputdir
INPUT_DIR="/work/data/inputs"
OUTPUT_DIR="/work/data/outputs"

nvidia-docker run --rm \
    -v $inputdir:/work/data/inputs \
    -v $outputdir:/work/data/outputs \
    $dockerimage \
    /bin/bash -c "echo running docker; \
    python ${inference_script} \
        -i ${INPUT_DIR} \
        -o ${OUTPUT_DIR} \
    "
