import argparse
import numpy as np
from time import time
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import logging
import cv2
import h5py
import json

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoDKeyPoints:
	"""
	Class for point clouds Iterative Closest Point (ICP) implementation with
	open3d.
	"""
	coco_pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],
					   [6,7],[1,8],[8,9],[9,10],[1,11],[11,12],
					   [12,13],[0,14],[0,15],[14,16],[15,17]]
	coco_key_points = [
			"nose", "left_eye", "right_eye", "left_ear", "right_ear",
			"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
			"left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
			"right_knee", "left_ankle", "right_ankle"]
	npoints = 18
	
	def __init__(self, config={
		"proto": "openpose/models/pose/coco/pose_deploy_linevec.prototxt",
		"weight": "openpose/models/pose/coco/pose_iter_440000.caffemodel"}):
		"""
		Initialization.
		
		Parameters:
			filenames, list of str, full path to the filnames of the input point
				clouds
		CoCo key points:
			"nose", "left_eye", "right_eye", "left_ear", "right_ear",
			"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
			"left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
			"right_knee", "left_ankle", "right_ankle"
		"""
		self.config = config

	def _load_network(self):
		"""
		load the network if it is needed for inference
		"""
		assert "proto" in self.config, "proto file not found!"
		assert "weight" in self.config, "weight file not found!"
		self.prob_threshold = self.config.get("prob_threshold", 0.5)
		# Read the network into Memory
		self.network = cv2.dnn.readNetFromCaffe(self.config["proto"], self.config["weight"])
		logger.info("Load proto file: %s" % (self.config["proto"]))
		logger.info("Load weight file: %s" % (self.config["weight"]))

	@staticmethod
	def _get_frame(image_path):
		assert os.path.exists(image_path), "%s not found" % image_path
		return cv2.imread(image_path)

	def inference(self, image, out_path=None):
		"""
		Make inference of the 2d key points model
		Parameters:
			image: str of image path, or np.array as image
		"""
		if not hasattr(self, "network"):
			self._load_network()

		self.frame = self._get_frame(image) if isinstance(image, str) else image
		height, width = self.frame.shape[0], self.frame.shape[1]
		print("height", height, "width", width)
		inputblob = cv2.dnn.blobFromImage(
					self.frame, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False)
		self.network.setInput(inputblob)
		output = self.network.forward()
		self.keypoints = self._get_keypoints(output, width, height)
		logger.info("Detected keypoints: %s" % str(self.keypoints))
		kpdict = {key: p for key, p in zip(self.coco_key_points, self.keypoints) if p is not None}
		if out_path is not None:
			outdir = os.path.dirname(out_path)
			os.makedirs(outdir, exist_ok=True)
			with open(out_path, "w") as f:
				json.dump(kpdict, f, indent=4 )
			logger.info("output file saved to: %s" % out_path)
		logger.info("detected keypoints: %s" % str(kpdict))
		return kpdict

	def _get_keypoints(self, output, width, height):
		print("kp output shape", output.shape)
		out_h = output.shape[2]
		out_w = output.shape[3]
		print("output height", out_h, "width", out_w)

		# Empty list to store the detected keypoints
		points = []
		for i in range(self.npoints):
			# confidence map of corresponding body's part.
			probMap = output[0, i, :, :]
			# Find global maxima of the probMap.
			minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

			# Scale the point to fit on the original image
			x = (width * point[0]) / out_w
			y = (height * point[1]) / out_h

			print ("index", i, "prob", prob)
			if prob > self.prob_threshold:
				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))
			else:
				points.append(None)
		return points

	def draw(self, outname):

		framecopy = np.copy(self.frame)
		for idx, p in enumerate(self.keypoints):
			if p is None: continue
			cv2.circle(framecopy, p, 15, (0, 255, 255),
                            thickness=-1, lineType=cv2.FILLED)
			cv2.putText(framecopy, "{}".format(idx),
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.4,
							(0, 0, 255), 3, lineType=cv2.LINE_AA)

		for pair in self.coco_pose_pairs:
			partA = pair[0]
			partB = pair[1]

			if self.keypoints[partA] and self.keypoints[partB]:
				cv2.line(framecopy, self.keypoints[partA], self.keypoints[partB], (0, 255, 0), 3)
		cv2.imwrite(outname)
		logger.info("Save rgbd image: %s" % outname)

	@staticmethod
	def load_h5(filename):
		"""
		Load key points from 2d key points module in the pipeline
		There are 17 key points:
			['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
			'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
			'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
			'right_knee', 'left_ankle', 'right_ankle']
		"""
		assert os.path.exists(filename), "%s not found" % filename
		kpdict = {}
		with h5py.File(filename, "r") as h5:
			# key points names
			names = [name.decode('ascii') for name in np.array(h5["info"]["twoD_keypoints"])]
			for name in h5["data"]:
				# rename
				if name[0:5] != "frame": continue
				idx = int(name[5:])
				# np.array(h5["data"][name]) has 17 x 3 array... with float
				# need to save 17 x 2 with int
				kps = []
				for p in np.array(h5["data"][name]):
					kp = None if np.any(np.isnan(p)) else kps.append([int(p[0]), int(p[1])])
					kps.append(kp)
				kpdict["FRAME%04d" % idx] = {name: kps[i] for i, name in enumerate(names)}
				logger.info("loaded kp for %s" % name)
		return kpdict

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input", "-i", default=None, type=str, required=True,
		help="Your input point cloud file.",
	)
	parser.add_argument(
		"--outdir", "-o", default="outputs/keypoints", type=str, required=False,
		help="Your output folder.",
	)

	args = parser.parse_args()
	logger.info("Reading input file: %s" % args.input)
	sc = TwoDKeyPoints()
	basename = os.path.splitext(os.path.basename(args.input))[0]
	print("basename", basename)
	sc.inference(args.input, os.path.join(args.outdir, basename+"_keypoints.json"))
	# sc.draw(os.path.join(args.outdir, basename+"_out.png"))
	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	"""
	Execute example:
		python3 pointcloudstransformation.py \
			-i outputs/point_clouds/FRAME0000.ply \
			-o /path/to/output/output_transformed
	"""
	main()
