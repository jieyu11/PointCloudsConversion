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


class PointCloudsICP:
	"""
	Class for point clouds Iterative Closest Point (ICP) implementation with
	open3d.
	"""
	def __init__(self, filenames):
		"""
		Initialization.
		
		Parameters:
			filenames, list of str, full path to the filnames of the input point
				clouds
		"""
		assert isinstance(filenames, list) and len(filenames) >= 2, "min 2 inputs"
		self.nframes = len(filenames)
		self.pcds = [o3d.io.read_point_cloud(fname) for fname in filenames]
		self.inputdir = os.path.dirname(filenames[0])
		self.inputnames = [os.path.splitext(os.path.basename(fname))[0] for fname in filenames]
		self.icp_transformations = None
		logger.info("%d input names: %s" % (self.nframes, str(self.inputnames)))

	def even_rotation(self):
		"""
		Expecting evenly rotated frames. For example, if there are 4 frames in the input, then,
		each is rotated by 90 degrees after every previous one
		# idx 25, 80, 125, 190
		"""
		center_0 = self.pcds[0].get_center()
		for idx, pcd in enumerate(self.pcds):
			if idx == 0: continue
			angle = (0, idx * 2 * np.pi / self.nframes, 0)
			center = pcd.get_center()
			pcd = self.rotate(pcd, angle, center)
			# move the center of the frame towards the 0-th frame
			trans_m = tuple([a-b for a,b in zip(center_0, center)])
			print(idx, "rotate angle: ", angle, " trans ", trans_m)
			self.pcds[idx] = pcd.translate(trans_m)

	def icp(self, mode="color"):
		"""
		Applying icp with mode=color or geo
		"""
		assert mode in ["color", "geo"], "mode=%s not known" % mode
		print("Apply point-to-point ICP")
		threshold = 0.02
		self.icp_transformations = []
		# start from unit matrix
		for idx in range(self.nframes - 1):
			# transformation from idx+1 to idx
			source = self.pcds[idx+1]
			target = self.pcds[idx]
			trans_init = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
			if mode == "color":
				logger.info("Making color icp")
				reg_p2p = self._color_icp_transformation(source, target, threshold, trans_init)
			else:
				logger.info("Making geo icp")
				reg_p2p = self._geo_icp_transformation(source, target, threshold, trans_init)

			self.icp_transformations.append(reg_p2p.transformation)
			logger.info("Transformation from %s to %s is %s " % (
				self.inputnames[idx+1], self.inputnames[idx], str(reg_p2p.transformation)
			))
		# draw_registration_result(source, target, reg_p2p.transformation)
	def _color_icp_transformation(self, source_pcd, target_pcd, radius, trans_curr, niter=50):
		"""
		ref: http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
		"""
		source_pcd.estimate_normals(
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
		target_pcd.estimate_normals(
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

		result_icp = o3d.pipelines.registration.registration_colored_icp(
			source_pcd, target_pcd, radius, trans_curr,
			o3d.pipelines.registration.TransformationEstimationForColoredICP(),
			o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
														  relative_rmse=1e-6,
														  max_iteration=niter))
		return result_icp

	def _geo_icp_transformation(source_pcd, target_pcd, radius, trans_curr, niter=50):
		"""
		ref: http://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
		"""
		result_icp = o3d.pipelines.registration.registration_icp(
  				  source_pcd, target_pcd, radius, trans_curr,
					o3d.pipelines.registration.TransformationEstimationPointToPoint(),
					o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=niter)
					)
		return result_icp
	
	def clear_ground(self):
		"""
		Ground is defined around the minimum on Y axis.
		"""
		for idx, pcd in enumerate(self.pcds):
			pcdarr = np.asarray(pcd.points)
			pcdcol = np.asarray(pcd.colors)
			# use color to select the points, if the points are having
			# the same value for color in x, y, z, e.g. values: [0.49019608 0.49019608 0.49019608]
			# then remove them
			idx_select = np.array([i for i, v in enumerate(pcdcol) if v[0]==v[1]==v[2]])
			print("Center in groud: ", pcd, pcd.get_center())
			self.pcds[idx] = pcd.select_by_index(idx_select, invert = True)
			print("Center transformed: ", self.pcds[idx], self.pcds[idx].get_center())
			# # y clear bottom 0.25
			# # x clear 0.1 both sides
			# ymin5 = np.quantile(pcdarr[:, 0], 0.1)
			# idx_select = np.array([i for i, v in enumerate(pcdarr) if v[0]<ymin5])
			# print(" y 10% value: ", ymin5)
			# self.pcds[idx] = pcd.select_by_index(idx_select)
			# print("colors", self.pcds[idx].colors, np.array(self.pcds[idx].colors)[0:100])
	
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

	def rotate(self, pcd, angle=(np.pi, 0, 0), center=(0, 0, 0)):
		"""
		Rotate the point cloud with the given angle
		"""
		print("Center before rotation: ", pcd, pcd.get_center())
		R = pcd.get_rotation_matrix_from_xyz(angle)
		pcd.rotate(R, center=center)
		print("Center transformed: ", pcd, pcd.get_center())
		return pcd
	
	def flip(self, mode="left/right"):
		assert mode in ["left/right", "up/down"], "Mode: %s not known." % mode
		# left-right
		matrix = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
		# up-down
		if mode == "up/down":
			matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
		for pcd in self.pcds:
			print("Center in flip: ", pcd, pcd.get_center())
			pcd.transform(matrix)
			print("Center transformed: ", pcd, pcd.get_center())
	
	def draw(self, outdir):
		for idx, pcd in enumerate(self.pcds):
			plt.figure(figsize=(8, 6))
			o3d.visualization.draw([pcd])
			# logger.info("Downsample the point cloud with a voxel of 0.02")
			# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
			# o3d.visualization.draw([voxel_down_pcd])
			outname = os.path.join(outdir, self.inputnames[idx]+".png")
			plt.savefig(outname)
			logger.info("Save %s" % outname)

	def save(self, outdir):
		for idx, pcd in enumerate(self.pcds):
			outname = os.path.join(outdir, self.inputnames[idx]+".ply")
			o3d.io.write_point_cloud(outname, pcd)
			print("idx=",idx, "save pcd", pcd, pcd.get_center())
			logger.info("Save point cloud: %s" % outname)

			if idx > 0 and self.icp_transformations is not None:
				# apply transformations on to frame idx+1
				# in the end, every frame should look reasonable in the original frame
				# of the idx = 0 frame
				for j in range(idx):
					pcd = pcd.transform(self.icp_transformations[j])
					print("After tran", j, " Center: ", pcd, pcd.get_center())
				outname = os.path.join(outdir, self.inputnames[idx]+"_transformed.ply")
				o3d.io.write_point_cloud(outname, pcd)
				print("save transformed pcd", pcd)
				logger.info("Save point cloud: %s" % outname)


def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--inputs", "-i", default=None, type=str, required=True,
		nargs="+",
		help="Your input point cloud file.",
	)
	parser.add_argument(
		"--output", "-o", default="outputs/icp", type=str, required=False,
		help="Your output folder.",
	)

	args = parser.parse_args()
	for filename in args.inputs:
		assert os.path.exists(filename), "%s doesn't exit!" % filename
	logger.info("Reading input files: %s" % str(args.inputs))
	sc = PointCloudsICP(args.inputs)
	# sc.clear_ground()
	# sc.rotate(angle=(0, np.pi, 0))
	# sc.flip()
	# sc.outlier_removal()
	sc.icp()
	# sc.even_rotation()
	os.makedirs(args.output, exist_ok=True)
	# sc.draw(args.output)
	sc.save(args.output)

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
