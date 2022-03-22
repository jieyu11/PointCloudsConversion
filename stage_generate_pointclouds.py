import argparse
from time import time
from datetime import timedelta
import logging
from depth2pointclouds_stageh5 import TrueDepthToPointClouds
import os

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--h5", "-h5", default=None, type=str, required=True,
		help="Your input h5 file.",
	)
	parser.add_argument(
		"--kph5", "-kp", default=None, type=str, required=True,
		help="Your keypoint h5 file from pipeline running.",
	)
	parser.add_argument(
		"--output", "-o", default="outputs", type=str, required=False,
		help="Your output directory",
	)
	parser.add_argument(
		"--camera", "-c", default="CAM0", type=str, required=False,
		help="Your camera to read, default: CAM0",
	)
	parser.add_argument(
		"--nframes", "-n", default=10000, type=int, required=False,
		help="Your max number of frames to read.",
	)
	parser.add_argument(
		"--ndraw", "-nd", default=10, type=int, required=False,
		help="Your max number of frames to draw.",
	)
	parser.add_argument(
		"--mask", "-m", default=None, type=str, required=False,
		help="Your images' mask folder obtained from background segmenation.",
	)

	args = parser.parse_args()
	logger.info("Reading h5 file: %s" % args.h5)
	sc = TrueDepthToPointClouds(args.h5, config={
		"camera": args.camera,
		"nframes": args.nframes,
		"mask": args.mask
		})
	sc.get_keypoints(args.kph5)
	sc.convert_keypoints_to_3d()
	sc.convert_keypoints_to_pointcloud(outdir=os.path.join(args.output, "keypoints"))
	for mode in ["color", "depth"]:
		# do mask images before cropping
		sc.mask_images(mode=mode)
		# sc.crop_images(mode=mode)
	# sc.crop_around_keypoints()
	sc.create_rgbd_images()
	sc.create_point_clouds(outdir=os.path.join(args.output, "point_clouds"))
	sc.draw_rgbd(os.path.join(args.output, "rgbd"), args.ndraw)


	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	"""
	Execute example:
		python3 stage_generate_pointclouds.py \
			-h5 ../r0-2022-01-25-ITJohnFrontV2-018.h5 \
			-o /path/to/output/output_pointclouds/ \
			-n 10 -nd 10 -m masks/deeplabraw
	"""
	main()
