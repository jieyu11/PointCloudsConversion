import argparse
from time import time
from datetime import timedelta
import logging
from depth2pointclouds import DepthToPointClouds
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

	args = parser.parse_args()
	logger.info("Reading h5 file: %s" % args.h5)
	sc = DepthToPointClouds(args.h5, config={
		"camera": args.camera,
		"nframes": args.nframes,
		})
	sc.draw_color(os.path.join(args.output, "color"), args.ndraw)

	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	main()
