# Depth Map from Stage Recorded H5 to Point Clouds
The goal of the developed package below is to convert depth map images
to point clouds with tools like `open3d`. The tested input depth map images are initially recorded by the stage system. An example of such
h5 file can be found in the nfs system:

```
192.168.1.74:/groups1/3DSelfie/sample_videos/DepthVideos/r0-2022-01-25-ITJohnFrontV2-018.h5
```

## Requirements
It is tested that this conversion is sufficiently fast while running
with CPU on a laptop. The packages like:
```
open3d
numpy
cv2
h5py
```
are required. The versions of the modules from a tested system with
python version 3.9.5 are listed below:
```
open3d==0.14.1
opencv-python==4.5.1.48
opencv-python-headless==4.4.0.46
numpy==1.19.5     
h5py==3.1.0
```
Though, it is expected that any version of python 3+ with the corresponding modules should work.

## Generate Segmentations
To clean up the depth map and remove the background other than the
human, it is better to apply a segmentation to remove the background.

To do so, the docker image of `bgsegmentation_deeplab_retrain` is
required, which should have been built with the code execution of
`tron/setup/gepetto_build.sh` during modules building.

To generate the images from the h5 file:
```
	python3 stage_generate_images.py -h5 /path/to/h5/file.h5 \
	-o /path/to/dir/output/color/ -n 300 -nd 300
```
Note: `-n` defines how many frames to be read from the h5 file and
`-nd` defines the number of frames to draw and saved in the output
folder.

The run the segmentation generation with the following script:
```
run_standalone_deeplabv3.sh
```
with modifications of the parameters in the file. Make sure to use
the color images generated above to the obtain the segmentations
of human vs background.

## Generate Point Clouds
Now we are ready to genarate the point clouds with the depth
map as well as the masks (segmentations of human):

```
python3 stage_generate_pointclouds.py -h5 /path/to/h5/file.h5 \
	-m /path/dir/to/input/mask/ \
	-o /path/dir/to/output/point-clouds/ -n 300 -nd 10 
```

## Read SMPL Model
To read SMPL based model, install related modules listed in:
https://github.com/vchoutas/smplx
