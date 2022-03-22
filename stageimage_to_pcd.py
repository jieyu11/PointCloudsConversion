import argparse
from time import time
from datetime import timedelta
import logging
from depth2pointclouds_disparity import DisparityDepthToPointClouds
import os

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--output", "-o", default="outputs/stageimages", type=str, required=False,
		help="Your output directory",
	)
	parser.add_argument(
		"--camera-config", "-c", default=None, type=str, required=True,
		help="Your camera to config file.",
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
		"--maskdir", "-m", default=None, type=str, required=True,
		help="Your images mask folder obtained from background segmenation.",
	)
	parser.add_argument(
		"--imagedir", "-i", default=None, type=str, required=True,
		help="Your rgb images folder.",
	)
	parser.add_argument(
		"--depthdir", "-d", default=None, type=str, required=True,
		help="Your depth images folder.",
	)
	parser.add_argument(
		"--side", "-s", default=None, type=str, required=True,
		help="Your side of the image either left or right for camera calib.",
	)
	parser.add_argument(
		"--qmatrix", "-qm", default=None, type=str, required=False,
		help="Your side of the image either left or right for camera calib.",
	)

	args = parser.parse_args()
	print("config", args.camera_config)
	sc = DisparityDepthToPointClouds(config={
		"camera_calib": args.camera_config,
		"nframes": args.nframes,
		"maskdir": args.maskdir,
		"imagedir": args.imagedir,
		"depthdir": args.depthdir,
		"qmatrixdir": args.qmatrix,
		"side": args.side,
		})
	for mode in ["color", "depth"]:
		# do mask images before cropping
		sc.mask_images(mode=mode)
		sc.crop_images(mode=mode)
	sc.create_rgbd_images()
	sc.create_point_clouds(outdir=os.path.join(args.output, "point_clouds"))
	sc.draw_rgbd(os.path.join(args.output, "rgbd"), args.ndraw)


	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))

if __name__ == "__main__":
	"""
	Execute example:
		python3 stageimage_to_pcd.py \
			-c images/BobAPose/CalibratedStage.json \
			-i images/BobAPose/rgb_left \
			-m images/BobAPose/mask_left \
			-d images/BobAPose/depth \
			-qm images/BobAPose/q_matrix \
			-n 10 -nd 10
	"""
	main()
