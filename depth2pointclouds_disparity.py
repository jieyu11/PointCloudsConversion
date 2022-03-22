import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import logging
from stagecalibration import StageCalibration
from utils import convert_image_color, get_boundary

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class DisparityDepthToPointClouds:
	def __init__(self, config):
		"""
		Initialization.
		
		Parameters:
			config, dict, parameters
				"imagedir:, str, full path to folder containing rgb images
				"depthdir:, str, full path to folder containing depth images
				"camera", str, (default: CAM0), name of the camera used to retrieve
					depth and color images.
				"nframes", int, (default: 10000), number of frames to be read from file.
				"maskdir", str, folder containing the image masks generated with segmentation
					algorithms like DeepLabV3.
		"""
		self.nframes = config.get("nframes", 10000)
		assert "imagedir" in config and "depthdir" in config
		self._load_image_dir(imagedir=config["imagedir"], depthdir=config["depthdir"])
		assert "camera_calib" in config, "must have camera_calib file name in config"
		self.camera_calib = StageCalibration(config["camera_calib"]).get_config()
		assert "maskdir" in config, "generate mask first"
		self.frames_masks = self._get_masks(config["maskdir"])
		assert "side" in config and config["side"] in ["left", "right"]
		self.side = config["side"]
		if "qmatrixdir" in config and config["qmatrixdir"] is not None:
			self.qmatrix = self._load_qmatrix_dir(config["qmatrixdir"])
		logger.info("RGB frames: %s" % str(self.frames_color.keys()))

		# reshape the depth dimension to match the one from color
		# convert -1 values to np.nan
		for name, frame_depth in self.frames_depth.items():
			# depth with values below 0 is ill defined. Use np.nan instead
			# np.nan is float, so need to use float type
			frame_depth = frame_depth.astype("float")
			frame_depth[frame_depth < 0] = np.nan
			self.frames_depth[name] = frame_depth
			# self.frames_depth[name] = cv2.resize(frame_depth, (self.frame_width, self.frame_height))
		self.bounds = get_boundary([f for _, f in self.frames_masks.items()])
		logger.info("Boundary of the frames: %s" % self.bounds)

		for name, frame_depth in self.frames_depth.items():
			print("depth shape", frame_depth.shape)
			print("depth ", frame_depth)
			print("unique ", np.unique(frame_depth))
			break

	def _load_image_dir(self, imagedir, depthdir):
		imagenames = os.listdir(imagedir)
		self.frames_color, self.frames_depth = {}, {}
		for imgname in imagenames:
			# depth image should have the same file name as rgb image
			assert os.path.exists(os.path.join(depthdir, imgname)), "depth not found"
			basename = os.path.splitext(imgname)[0]
			self.frames_color[basename] = cv2.imread(os.path.join(imagedir, imgname))
			# self.frames_depth[basename] = cv2.imread(os.path.join(depthdir, imgname))
			# read depth with gray scale
			# self.frames_depth[basename] = cv2.imread(os.path.join(depthdir, imgname))
			self.frames_depth[basename] = cv2.imread(os.path.join(depthdir, imgname),
                                            cv2.IMREAD_UNCHANGED).astype("float") / 256.0
                                            # cv2.IMREAD_GRAYSCALE)
		logger.info("N frames, color: %d; depth: %d" % (len(self.frames_color),
                                                  len(self.frames_depth)))

	def pcd_outlier_removal(self, pcd, mode="statistical", config={}):
		assert mode in ["statistical", "radius"], "Mode: %s not found" % mode
		if mode == "statistical":
			logger.info("Statistical oulier removal")
			nb_neighbours = config.get("nb_neighbours", 20)
			std_radio = config.get("std_ratio", 2.0)
			cl, ind = pcd.remove_statistical_outlier(
				nb_neighbors=nb_neighbours, std_ratio=std_radio)
			pcd = pcd.select_by_index(ind)
		else:
			logger.info("Radius oulier removal")
			nb_points = config.get("nb_points", 16)
			radius = config.get("radius", 0.05)
			cl, ind = pcd.remove_radius_outlier(
				nb_points=nb_points, radius=radius)
			pcd = pcd.select_by_index(ind)
		print("PC after removal ", pcd)
		return pcd
	
	def _load_qmatrix_dir(self, qmatrixdir):
		jsonnames = os.listdir(qmatrixdir)
		self.frames_qmatrix = {}
		for name in jsonnames:
			filename = os.path.join(qmatrixdir, name)
			assert os.path.exists(filename)
			basename = os.path.splitext(name)[0]
			with open(filename) as f:
				# json file contains:
				# {'Q': [
				# [1.0, 0.0, 0.0, -1819.5047302246094],
				# [0.0, 1.0, 0.0, -1082.4932098388672],
				# [0.0, 0.0, 0.0, 4155.094444363626],
				# [0.0, 0.0, 19.551827028852372, -0.0]
				# ]}
				print("filename", filename)
				data = json.load(f)
				print("Q data", data)
				# convert Q matrix to: cv2.Umat() with np.float32
				self.frames_qmatrix[basename] = np.float32(data["Q"])
			print("qmatrix", name, self.frames_qmatrix[basename])
		logger.info("N frames q matrix: %d" % len(self.frames_qmatrix))

	def create_point_clouds(self, outdir=None):
		"""
		Create point clouds with q-matrix or rgbd images
		"""
		if outdir is not None:
			os.makedirs(outdir, exist_ok=True)

		if hasattr(self, "frames_qmatrix"):
			logger.info("creating point clouds with depth and Q matrix!")
			self._create_point_clouds_qmatrix(outdir)
		elif hasattr(self, 'frames_rgbd'):
			logger.info("creating point clouds with rgbd images and intrinsic matrix!")
			self._create_point_clouds_rgbd(outdir)
		else:
			logger.error("no q-matrix or rgbd found!!! Cannot make point clouds!")

	def pcd_clear_ground(self, pcd):
		"""
		Ground is defined around the minimum on Y axis.
		"""
		pcdarr = np.asarray(pcd.points)
		pcdcol = np.asarray(pcd.colors)
		# use color to select the points, if the points are having
		# the same value for color in x, y, z, e.g. values: [0.49019608 0.49019608 0.49019608]
		# then remove them
		idx_select = np.array([i for i, v in enumerate(pcdcol) if v[0]==v[1]==v[2]])
		print("Center in groud: ", pcd, pcd.get_center())
		pcd = pcd.select_by_index(idx_select, invert = True)
		print("Center transformed: ", pcd, pcd.get_center())
		return pcd

	def _create_point_clouds_qmatrix(self, outdir=None):
		"""
		Create point clouds from RGBD images from qmatrix.
		"""
		self.point_clouds = {}
		for name, frame_color in self.frames_color.items():
			assert name in self.frames_qmatrix, "Frame %s not found in qmatrix" % name
			assert name in self.frames_depth, "Frame %s not found in depth" % name
			print("q matrix type", type(self.frames_qmatrix[name]))
			print("q matrix", self.frames_qmatrix[name])
			points = cv2.reprojectImageTo3D(
				# cv2.cvtColor(self.frames_depth[name], cv2.COLOR_BGR2GRAY),
				np.float32(self.frames_depth[name]),
				self.frames_qmatrix[name]
				)
			pcd = o3d.geometry.PointCloud(points=points)
			pcd.colors = frame_color
			# Flip it, otherwise the pointcloud will be upside down
			pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			pcd = self.pcd_clear_ground(pcd)
			self.point_clouds[name] = pcd
			if outdir is not None:
				outname = os.path.join(outdir, name+".ply")
				o3d.io.write_point_cloud(outname, pcd)
				logger.info("Point clouds saved to: %s" % outname)
		logger.info("Number of point cloud files generated: %d" % len(self.point_clouds))	

	def _create_point_clouds_rgbd(self, outdir=None):
		"""
		Create point clouds from RGBD images from create_rgbd_images() function.
		"""
		self.point_clouds = {}
		for name, frame_rgbd in self.frames_rgbd.items():
			# name like: rig12_r04-sd02.png and extract "rig12"
			rigid = [n for n in name.split("_") if "rig" in n][0]
			logger.info("Got rig id: %s" % rigid)
			assert rigid in self.camera_calib, "rig: %s is not in calibration" % rigid
			cam = o3d.camera.PinholeCameraIntrinsic(
				width=self.camera_calib[rigid]["width"][self.side],
				height=self.camera_calib[rigid]["height"][self.side],
                # "intrinsics": [ 2084.2573717065397, 0.0,
                # 970.1369247436523, 0.0, 2084.2573717065397,
                # 545.4283218383789, 0.0, 0.0, 1.0 ]
				# Intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
				fx=self.camera_calib[rigid]["intrinsics"][self.side][0],
				fy=self.camera_calib[rigid]["intrinsics"][self.side][4],
				cx=self.camera_calib[rigid]["intrinsics"][self.side][2],
				cy=self.camera_calib[rigid]["intrinsics"][self.side][5]
				)
			pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			    frame_rgbd, cam)
			# Flip it, otherwise the pointcloud will be upside down
			pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			pcd = self.pcd_clear_ground(pcd)
			pcd = self.pcd_outlier_removal(pcd, mode="statistical")
			pcd = self.pcd_outlier_removal(pcd, mode="radius")
			self.point_clouds[name] = pcd
			if outdir is not None:
				outname = os.path.join(outdir, name+".ply")
				o3d.io.write_point_cloud(outname, pcd)
				logger.info("Point clouds saved to: %s" % outname)
		logger.info("Number of point cloud files generated: %d" % len(self.point_clouds))	

	def create_rgbd_images(self):
		self.frames_rgbd = {}
		for name, frame_color in self.frames_color.items():
			frame_depth = self.frames_depth[name]
			self.frames_rgbd[name] = o3d.geometry.RGBDImage.create_from_color_and_depth(
			    o3d.geometry.Image(frame_color.astype(np.uint8)),
                o3d.geometry.Image(frame_depth.astype(np.float32)),
                # depth_scale = 10, # default 1000
                # depth_trunc=.03, # default 3
				convert_rgb_to_intensity = False
				)
		logger.info("Number of RGBD images: %d" % len(self.frames_rgbd))

	def mask_images(self, mode="color"):
		"""
		Draw the number of color frames
		"""
		assert mode in ["color", "depth"], "mode is color or depth"
		assert self.frames_masks is not None, "to make masked images, need to have mask inputs"
		frames_dict = self.frames_color if mode == "color" else self.frames_depth
		for name, frame in frames_dict.items():
			assert name in self.frames_masks, "%s not found in masks" % name
			if mode == "color":
				# in mask images [0, 0, 0] is background [255,255,255] is person
				frame = convert_image_color(
						frame, [0, 0, 0], [125, 125, 125],
						src_img=self.frames_masks[name])
			else:
				# use np.nan to mask the background depth pixels
				frame[~self.frames_masks[name]] = np.nan 
			# update the frame
			frames_dict[name] = frame
		logger.info("Number of images masked for %s: %d" % (mode, len(frames_dict)))
	
	def crop_images(self, mode="color"):
		"""
		Crop the original sized image to have only the part of the human
		"""
		assert mode in ["color", "depth"], "mode is color or depth"
		if self.bounds is None:
			logger.error("Bound values are not found.")
			return
		frames_dict = self.frames_color if mode == "color" else self.frames_depth
		for name, frame in frames_dict.items():
			x0, x1, y0, y1 = self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3]
			frame = frame[x0:x1, y0:y1, :] if mode == "color" else frame[x0:x1, y0:y1]
			frames_dict[name] = frame
		logger.info("Number of images cropped for %s: %d" % (mode, len(frames_dict)))

	def draw_rgbd(self, outdir, nframes=10):
		"""
		Draw RGB and depth side by side.
		"""
		os.makedirs(outdir, exist_ok=True)
		for name, frame in self.frames_rgbd.items():
			if nframes <= 0: break
			plt.figure(figsize=(8, 6))
			plt.subplot(1, 2, 1)
			plt.title('%s (color)' % name)
			plt.imshow(frame.color)
			plt.xticks([]), plt.yticks([])
			plt.subplot(1, 2, 2)
			plt.title('%s (depth)' % name)
			plt.imshow(frame.depth)
			plt.xticks([]), plt.yticks([])
			plt.tight_layout()
			outname = os.path.join(outdir, name+".png")
			plt.savefig(outname)
			plt.clf()
			logger.info("Save rgbd image: %s" % outname)

			nframes -= 1

	def draw_color(self, outdir, nframes=10, masked=True):
		"""
		Draw the number of color frames
		"""
		os.makedirs(outdir, exist_ok=True)
		for name, frame in self.frames_color.items():
			if nframes <= 0: break
			outname = os.path.join(outdir, name+".png")

			if masked and self.frames_masks:
				assert name in self.frames_masks, "%s not found in masks" % name
				# in mask images [0, 0, 0] is background [255,255,255] is person
				frame = convert_image_color(
					frame, [0, 0, 0], [125, 125, 125],
					src_img=self.frames_masks[name])
			if self.bounds is not None:
				x0, x1, y0, y1 = self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3]
				frame = frame[x0:x1, y0:y1, :]
			logger.info("Saving color frame: %s" % outname)
			cv2.imwrite(outname, frame)
			nframes -= 1

	def draw_depth(self, outdir, nframes=10):
		"""
		Draw the number of color frames
		"""
		os.makedirs(outdir, exist_ok=True)
		for name, frame in self.frames_depth.items():
			if nframes <= 0: break
			# mask nan
			frame[~self.frames_masks[name]] = np.nan 
			# frame = np.ma.array(frame, mask=(frame == np.nan))
			vmin, vmax = np.nanmin(frame), np.nanmax(frame)
			factor = 255. / (vmax - vmin)
			notnan = ~np.isnan(frame)

			# depth frame is in float type
			# convert values from min to max to 0 - 255
			frame[notnan] = (frame[notnan] - vmin) * factor

			# depth_convert = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
			depth_convert = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float)
			depth_convert[:, :, 0] = frame
			depth_convert[:, :, 1] = frame
			depth_convert[:, :, 2] = frame
			if self.bounds is not None:
				x0, x1, y0, y1 = self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3]
				depth_convert = depth_convert[x0:x1, y0:y1, :]
			outname = os.path.join(outdir, name+".png")
			logger.info("Saving depth frame: %s" % outname)
			cv2.imwrite(outname, depth_convert)
			nframes -= 1

	def _clean_depth_with_mask(self):
		"""
		Clean the depth image with mask input
		"""
		for name, frame in self.frames_depth.items():
			if name not in self.frames_masks:
				logger.warning("Frame %s has no mask!" % name)
				continue
			mask = self.frames_masks[name][:, :, 0]
			frame[mask==0] = np.nan
			self.frames_depth[name] = frame

	def _get_masks(self, mask_folder, keep_index=255):
		"""
		Get the masks for each image
		"""
		logger.info("Get masks from: %s" % mask_folder)
		frame_dict = {}
		for filename in os.listdir(mask_folder):
			image = cv2.imread(os.path.join(mask_folder, filename))
			# convert any color other than [keep_index]*3 to [0, 0, 0]
			image = convert_image_color(image, [keep_index]*3, [0, 0, 0], inverse=True)
			image = convert_image_color(image, [keep_index]*3, [255, 255, 255])
			name = os.path.splitext(filename)[0]
			frame_dict[name] = image
		logger.info("Got N frames for mask: %d" % len(frame_dict))
		return frame_dict
