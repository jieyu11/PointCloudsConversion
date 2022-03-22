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


class TwoDKeyPointsLoader:
	"""
	Class for point clouds Iterative Closest Point (ICP) implementation with
	open3d.
	"""
	# coco_pose_pairs = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],
	# 				   [6,7],[1,8],[8,9],[9,10],[1,11],[11,12],
	# 				   [12,13],[0,14],[0,15],[14,16],[15,17]]
	coco_pose_pairs = [[1,0], [2,0],[1,2], [1,3],[1,5],[2,4],[2,6],
						[5,6],[5,7],[6,8], [7,9], [8 , 10], 
						[5,11], [6,12], [11, 12], [11, 13], [12,14], [13,15], [14,16]]
	coco_key_points = [
			"nose", "left_eye", "right_eye", "left_ear", "right_ear", # 0 - 4
			"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", # 5-8
			"left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", # 9 - 13
			"right_knee", "left_ankle", "right_ankle"] # 14 - 16
	coco_pose_pair_dict = {
		"nose": ["left_eye", "right_eye"],
		"left_eye": ["left_ear", "right_eye"],
		"right_eye": ["right_ear"],
		"left_shoulder": ["left_elbow", "left_hip", "right_shoulder"],
		"right_shoulder": ["right_elbow", "right_hip"],
		"left_elbow": ["left_wrist"], 
		"right_elbow": ["right_wrist"], 
		"left_hip": ["left_knee", "right_hip"],
		"right_hip": ["right_knee"],
		"left_knee": ["left_ankle"],
		"right_knee": ["right_ankle"]
	}
	npoints = 18
	
	def __init__(self, kph5=None, frameh5=None):
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
		if kph5 is not None:
			self.key_points = self.load_kph5(kph5)
		if frameh5 is not None:
			self.frames = self.load_frameh5(frameh5)

	def draw(self, outdir):
		for name in self.frames:
			outname = os.path.join(outdir, name+".png")
			self.plot_one_frame(name, outname)

	def plot_one_frame(self, key, outname):

		frame = self.frames[key]
		keypoints = self._kplists[key]
		for idx, p in enumerate(keypoints):
			if p is None: continue
			p = tuple(p)
			cv2.circle(frame, p, 15, (0, 255, 255),
                            thickness=-1, lineType=cv2.FILLED)
			cv2.putText(frame, "{}".format(idx),
                            p, cv2.FONT_HERSHEY_SIMPLEX, 1.4,
							(0, 0, 255), 3, lineType=cv2.LINE_AA)

		for pair in self.coco_pose_pairs:
			partA = pair[0]
			partB = pair[1]
			if keypoints[partA] and keypoints[partB]:
				cv2.line(frame, tuple(keypoints[partA]), tuple(keypoints[partB]), (0, 255, 0), 3)

		cv2.imwrite(outname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		logger.info("Save rgbd image: %s" % outname)

	def load_kph5(self, filename):
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
		self._kplists = {}
		with h5py.File(filename, "r") as h5:
			# key points names
			names = [name.decode('ascii') for name in np.array(h5["info"]["twoD_keypoints"])]
			for name in h5["data"]:
				if name[0:5] != "frame": continue
				# frame index
				idx = int(name[5:])
				# np.array(h5["data"][name]) has 17 x 3 array... with float
				# need to save 17 x 2 with int
				kps = []
				for p in np.array(h5["data"][name]):
					kp = None if np.any(np.isnan(p)) else [int(p[0]), int(p[1])]
					kps.append(kp)
				kpdict["FRAME%04d" % idx] = {name: kps[i] for i, name in enumerate(names)}
				self._kplists["FRAME%04d" % idx] = kps
				logger.info("loaded kp for %s" % name)
				logger.info("Frame: %d, kp: %s" % (idx, str(kpdict["FRAME%04d" % idx])))
				logger.info("KPmax: %s" % str(np.max([k for k in kps if k], axis=0)))
		return kpdict

	def load_frameh5(self, filename, nframes=10000, camera="CAM0", mode="BGR"):
		"""
		Retrieve the color or depth frames
		"""
		assert mode in ["BGR", "Z"], "Frames are with mode=BGR(color) or Z(depth)"
		assert os.path.exists(filename), "Frame h5: %s not found" % filename
		frames_dict = {}
		with h5py.File(filename, "r") as h5:
			for key in h5:
				if "FRAME" not in key: continue
				if nframes <= 0: break
				frame = np.array(h5[key]["RAW"][camera][mode])
				# convert color to RGB
				if mode == "BGR":
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frames_dict[key] = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
				nframes -= 1
			logger.info("Total number of %s frames: %d" % (mode, len(frames_dict)))
		return frames_dict

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--keypoint-h5", "-k", default=None, type=str, required=True,
		help="Your key points h5 file.",
	)
	parser.add_argument(
		"--raw-h5", "-f", default=None, type=str, required=True,
		help="Your frames h5 file.",
	)
	parser.add_argument(
		"--outdir", "-o", default="outputs/keypoints", type=str, required=False,
		help="Your output folder.",
	)

	args = parser.parse_args()
	logger.info("Reading keypoints input file: %s" % args.keypoint_h5)
	logger.info("Reading frames input file: %s" % args.raw_h5)
	sc = TwoDKeyPointsLoader(args.keypoint_h5, args.raw_h5)
	os.makedirs(args.outdir, exist_ok=True)
	sc.draw(args.outdir)
	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	"""
	Execute example:
		python3 twodkeypoints_loader.py \
			-k TwoDKeyPoints.h5 \
			-f r0-2022-01-25-ITJohnFrontV2-018.h5 \
			-o /path/to/output/output_transformed
	"""
	main()
