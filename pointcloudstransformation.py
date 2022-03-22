import argparse
from time import time
from datetime import timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudsTransformation:
	def __init__(self, filename):
		"""
		Initialization.
		
		Parameters:
			filename, str, full path to the filname of the input point cloud
		"""
		self.pcd = o3d.io.read_point_cloud(filename)
		logger.info("Read input file: %s" % filename)
		print("PC", self.pcd)
		pcd_array = np.asarray(self.pcd.points)
		print("PC 10 points", pcd_array[0:10])
		idx_select = np.array([i for i, v in enumerate(pcd_array) if v[1]>0])
		print(np.median(pcd_array, axis=1))
		self.pcd = self.pcd.select_by_index(idx_select)
		print("PC selected ", self.pcd)
	
	def outlier_removal(self, mode="statistical", config={}):
		assert mode in ["statistical", "radius"], "Mode: %s not found" % mode
		if mode == "statistical":
			logger.info("Statistical oulier removal")
			nb_neighbours = config.get("nb_neighbours", 20)
			std_radio = config.get("std_ratio", 2.0)
			cl, ind = self.pcd.remove_statistical_outlier(
				nb_neighbors=nb_neighbours, std_ratio=std_radio)
			self.pcd = self.pcd.select_by_index(ind)
		else:
			logger.info("Radius oulier removal")
			nb_points = config.get("nb_points", 16)
			radius = config.get("radius", 0.05)
			cl, ind = self.pcd.remove_radius_outlier(
				nb_points=nb_points, radius=radius)
			self.pcd = self.pcd.select_by_index(ind)
		print("PC after removal ", self.pcd)

	def rotate(self, angle=(np.pi, 0, 0)):
		"""
		Rotate the point cloud with the given angle
		"""
		print("Center: ", self.pcd.get_center())
		R = self.pcd.get_rotation_matrix_from_xyz(angle)
		self.pcd.rotate(R, center=(0, 0, 0))
		print("Center 2: ", self.pcd.get_center())
	
	def flip(self, mode="left/right"):
		assert mode in ["left/right", "up/down"], "Mode: %s not known." % mode
		# left-right
		matrix = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
		# up-down
		if mode == "up/down":
			matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
		self.pcd.transform(matrix)
	
	def draw(self, output):
		plt.figure(figsize=(8, 6))
		o3d.visualization.draw([self.pcd])
		logger.info("Downsample the point cloud with a voxel of 0.02")
		voxel_down_pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
		o3d.visualization.draw([voxel_down_pcd])
		plt.savefig(output)

	def save(self, output):
		o3d.io.write_point_cloud(output, self.pcd)

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input", "-i", default=None, type=str, required=True,
		help="Your input point cloud file.",
	)
	parser.add_argument(
		"--output", "-o", default="outputs/transformed", type=str, required=False,
		help="Your output folder.",
	)

	args = parser.parse_args()
	assert os.path.exists(args.input), "input: %s doesn't exit!" % args.input
	logger.info("Reading input file: %s" % args.input)
	sc = PointCloudsTransformation(args.input)
	# sc.rotate(angle=(0, np.pi, 0))
	sc.flip()
	sc.outlier_removal()
	os.makedirs(args.output, exist_ok=True)
	basename = os.path.splitext(os.path.basename(args.input))[0]
	sc.draw(os.path.join(args.output, basename+".png"))
	sc.save(os.path.join(args.output, basename+".ply"))

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
