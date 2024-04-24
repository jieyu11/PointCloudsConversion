# Depth Map to Point Cloud Conversion

This package aims to convert depth map images into point clouds using tools like
`open3d`. The depth map images are typically recorded using the Stage system.

The example files provided with this package are in `.h5` format.

## Requirements

This conversion process has been tested for speed on a laptop CPU. The following
packages are required:

- `open3d`
- `numpy`
- `cv2`
- `h5py`

The tested versions of these modules on a system running Python 3.9.5 are as
follows:

```
open3d==0.14.1
opencv-python==4.5.1.48
opencv-python-headless==4.4.0.46
numpy==1.19.5     
h5py==3.1.0
```

However, it is expected that any version of Python 3+ with the corresponding
modules should work.

## Generating Segmentations

To enhance the depth map by removing background elements other than the human
subject, it's recommended to apply a segmentation process.

To do this, you'll need the Docker image `bgsegmentation_deeplab_retrain`, which
should be built using the code execution from `tron/setup/gepetto_build.sh`
during module building.

To generate images from the `.h5` file, use the following command:

```bash
python3 stage_generate_images.py -h5 /path/to/h5/file.h5 \
        -o /path/to/dir/output/color/ -n 300 -nd 300
```

- `-n`: Specifies the number of frames to read from the `.h5` file.
- `-nd`: Specifies the number of frames to draw and save in the output folder.

Then, run the segmentation generation with the provided script
`run_standalone_deeplabv3.sh`, ensuring you modify the parameters in the file.
Make sure to use the color images generated earlier to obtain segmentations of
human vs. background.

## Generating Point Clouds

With the depth map and human masks (segmentations) ready, you can generate point
clouds using the following command:

```bash
python3 stage_generate_pointclouds.py -h5 /path/to/h5/file.h5 \
        -m /path/dir/to/input/mask/ \
        -o /path/dir/to/output/point-clouds/ -n 300 -nd 10 
```

## Reading SMPL Model

To read SMPL-based models, install the related modules listed in [this
repository](https://github.com/vchoutas/smplx).

