import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import open3d as o3d
import logging
from twodkeypoints_loader import TwoDKeyPointsLoader
from utils import convert_image_color, get_boundary, pcd_clear_ground

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class TrueDepthToPointClouds:
	def __init__(self, filename, config):
		"""
		Initialization.
		
		Parameters:
			filename, str, full path to the filname of the input h5 file
			config, dict, parameters
				"camera", str, (default: CAM0), name of the camera used to retrieve
					depth and color images.
				"nframes", int, (default: 10000), number of frames to be read from file.
				"mask", str, folder containing the image masks generated with segmentation
					algorithms like DeepLabV3.
		"""
		self.camera = config.get("camera", "CAM0")
		self.nframes = config.get("nframes", 10000)
		with h5py.File(filename, "r") as h5:
			assert "INFO" in h5, "INFO is not found in %s" % filename
			assert self.camera in h5["INFO"], "%s not found" % self.camera
			self._get_camera_parameters(h5info=h5["INFO"])
			self.frames_color = self._get_frames(h5, self.nframes, mode="BGR")
			self.frames_depth = self._get_frames(h5, self.nframes, mode="Z")
			logger.info("N frames, color: %d; depth: %d" % (len(self.frames_color),
												   len(self.frames_depth)))
		# reshape the depth dimension to match the one from color
		# convert -1 values to np.nan
		for name, frame_depth in self.frames_depth.items():
			# depth with values below 0 is ill defined. Use np.nan instead
			# np.nan is float, so need to use float type
			frame_depth = frame_depth.astype("float")
			frame_depth[frame_depth < 0] = np.nan
			self.frames_depth[name] = cv2.resize(frame_depth, (self.frame_width, self.frame_height))
		self.frames_masks = None if "mask" not in config else self._get_masks(config["mask"])
		self.bounds = None
		if self.frames_masks is not None:
			self.bounds = get_boundary([f for _, f in self.frames_masks.items()])
		logger.info("Boundary of the frames: %s" % self.bounds)
		self._camera_intrinsic_matrix()
		# self._clean_depth_with_mask()
	
	def _camera_intrinsic_matrix(self):
		"""
		Set up camera intrinsic matrix
		"""
		# https://stackoverflow.com/questions/62809091/update-camera-intrinsic-parameter-in-open3d-python
		# Example CAM0 parameters: fx, fy, cx, cy = 930.20984 930.0072  636.0003  359.95996
		# Intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
		# cam.intrinsic_matrix =  [[930.2, 0.00, 636.0] , [0.00, 930.0, 359.9], [0.00, 0.00, 1.00]]
		# default options:
		# 	o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault
		# 	or PrimeSenseDefault ?
		try:
			self._cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
				width=self.frame_width, height=self.frame_height,
				fx=self.color_intrinsics[0], fy=self.color_intrinsics[1],
				cx=self.color_intrinsics[2], cy=self.color_intrinsics[3]
				)
			logger.info("Camera intrinsic: %s" % str(self._cam_intrinsic))
		except Exception:
			logger.error("Camera intrinsice matrix not loaded correctly.")
	
	def create_point_clouds(self, outdir=None):
		"""
		Create point clouds from RGBD images from create_rgbd_images() function.
		"""
		if not hasattr(self, 'frames_rgbd'):
			logger.error("Must have RGBD images to create point clouds.")
			return
		if outdir is not None:
			os.makedirs(outdir, exist_ok=True)
		self.point_clouds = {}
		for name, frame_rgbd in self.frames_rgbd.items():
			pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
				frame_rgbd, self._cam_intrinsic)

			# Flip it, otherwise the pointcloud will be upside down
			pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			# clear the ground in the pcd file
			pcd = pcd_clear_ground(pcd)
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
			# color_raw = o3d.io.read_image("images/stage_rgb/cam0_img_0000.png")
			# depth_raw = o3d.io.read_image("images/stage_depth/img_0000.png")
			# by default: convert_rgb_to_intensity = False
			self.frames_rgbd[name] = o3d.geometry.RGBDImage.create_from_color_and_depth(
				o3d.geometry.Image(frame_color.astype(np.uint8)),
				o3d.geometry.Image(frame_depth.astype(np.float32)),
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

	def _depth_to_3d(self, u, v, d, depth_scale=1000.):
		# for each frame, based on the x, y in 2D key points, get the z direction
		# calculation based on:
		# http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
		# #open3d.geometry.PointCloud.create_from_depth_image
		# Factory function to create a pointcloud from a depth image and a
		# camera. Given depth value d at (u, v) image coordinate, the
		# corresponding 3d point is :
		#     z = d / depth_scale
		#     x = (u - cx) * z / fx
		#     y = (v - cy) * z / fy
		# http://www.open3d.org/docs/release/python_api/open3d.camera.PinholeCameraIntrinsic.html
		if np.isnan(d):
			d = 3000.
		fx, fy = self._cam_intrinsic.get_focal_length()
		cx, cy = self._cam_intrinsic.get_principal_point()
		z = d / depth_scale
		x = (u - cx) * z / fx
		y = (v - cy) * z / fy
		print("DEPTH to 3D, input: ", u, v, d, " output ", x, y, z)
		return (x, y, z)

	def get_keypoints(self, h5name):
		"""
		Generate key points for each color image
		"""
		tk = TwoDKeyPointsLoader()
		self.frames_keypoints = tk.load_kph5(h5name)
		self.coco_pose_pair_dict = tk.coco_pose_pair_dict
		# self.frames_keypoints = {}
		# for name, frame in self.frames_color.items():
		# 	kpdict = tk.inference(frame)
		# 	self.frames_keypoints[name] = kpdict
		logger.info("Number of images for keypoint: %d" % len(self.frames_keypoints))

	def convert_keypoints_to_3d(self):
		"""
		Given the loaded keypoints, convert them into 3d based on the input of
		u, v, depth locations and the camera's intrinsic matrix.
		"""
		assert hasattr(self, "frames_keypoints"), "Run self.get_keypoints(h5name) frist!"
		self.frames_kp_3d = {}
		for name, kp_dict in self.frames_keypoints.items():
			# may have read fewer frames into frames_depth
			if name not in self.frames_depth: continue

			depth = self.frames_depth[name]
			# kp is a dict of keypoint_name: keypoint locations
			kp_points = {}
			for kpname, kploc in kp_dict.items():
				if kploc is None: continue
				#
				# width=self.frame_width, height=self.frame_height,
				# print("frame: ", name, "KP:", kpname, kploc)
				# u, v = kploc[0], kploc[1]
				# depth frame is in height, width order
				# kploc is in width, height order
				#
				v, u = kploc[0], kploc[1]
				d = depth[u][v]
				logger.info("Original u, v", u, v, "d", d)
				# original detected u, v may be inaccurate, which points to a depth
				# which can be nan or larger than it should be
				while np.isnan(d) or d < 1000. or d > 4000.:
					# move to left in width
					v -= 1
					if v < 0: break
					d = depth[u][v]
				logger.info("Modified u, v", u, v, "d", d)
				# note u, v is swapped
				# kp_points[kpname] = self._depth_to_3d(u, v, d)
				kp_points[kpname] = self._depth_to_3d(v, u, d)
			self.frames_kp_3d[name] = kp_points
			logger.info("Frame: %s key points in 3D: %s" % (name, str(kp_points)))

	def convert_keypoints_to_pointcloud(self, outdir):
		os.makedirs(outdir, exist_ok=True)
		# lines defined with: coco_pose_pair_dict in keypointsloader.
		for name, kp3d_dict in self.frames_kp_3d.items():
			print("frame", name)
			print("key points 3d", kp3d_dict)
			points = [p for k, p in kp3d_dict.items()]
			kpname = [k for k, p in kp3d_dict.items()]
			ktoidx = {p:idx for idx, p in enumerate(kpname)}
			# create lines
			lines = []
			for p1, p2list in self.coco_pose_pair_dict.items():
				if p1 not in kpname: continue
				for p2 in p2list:
					if p2 not in kpname: continue
					lines.append((ktoidx[p1], ktoidx[p2]))
			line_set = o3d.geometry.LineSet()
			line_set.points = o3d.utility.Vector3dVector(points)
			line_set.lines = o3d.utility.Vector2iVector(lines)
			# line_set.colors = o3d.utility.Vector3dVector([[255, 255, 0]] * len(lines))
			line_set.colors = o3d.utility.Vector3dVector([[255, 0, 255]] * len(lines))
			line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			outname = os.path.join(outdir, "lineset_"+name+".ply")
			o3d.io.write_line_set(outname, line_set)
			logger.info("Write line set to: %s" % outname)

			outname = os.path.join(outdir, "points_"+name+".ply")
			points = o3d.utility.Vector3dVector(points)
			pcd = o3d.geometry.PointCloud(points=points)
			pcd.colors = o3d.utility.Vector3dVector([[125, 125, 0]] * len(kpname))
			pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			o3d.io.write_point_cloud(outname, pcd)
			logger.info("Write KP point cloud to: %s" % outname)

	def crop_around_keypoints(self, key="nose", xpix=128, ypix=128):
		"""
		Generate key points for each color image
		"""
		assert hasattr(self, 'frames_keypoints'), "do not have keypoints for frames"
		counts = 0
		key_locations = []
		for name, frame in self.frames_color.items():
			kpdict = self.frames_keypoints[name]
			depth = self.frames_depth[name]
			masks = self.frames_masks[name]
			# looking for the key points with keys "nose", "left_eye",
			# "right_eye", "left_ear", "right_ear"
			# crop the initial image with 
			if key not in kpdict:
				logger.info("Key=%s not found in frame: %s" % (key, name))
				continue
			if kpdict[key] is None:
				kp = np.mean(key_locations, axis=1).astype(int)
				logger.info("%s location not found. Use average: %s" % (key, str(kp)))
				print("kp debug", kp)
			else:
				kp = kpdict[key]
				key_locations.append(kp)
				logger.info("Found %s at location: %s" % (key, str(kp)))
				
			xmin = max(kp[0] - xpix, 0)
			ymin = max(kp[1] - ypix, 0)
			xmax = kp[0] + xpix
			ymax = kp[1] + ypix
			self.frames_depth[name] = depth[xmin:xmax+1, ymin:ymax+1]
			self.frames_color[name] = frame[xmin:xmax+1, ymin:ymax+1, :]
			counts += 1

		logger.info("Number of images cropped for keypoint: %d" % counts)

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
		dege_dict = {}
		for filename in os.listdir(mask_folder):
			image = cv2.imread(os.path.join(mask_folder, filename))
			# convert any color other than [keep_index]*3 to [0, 0, 0]
			image = convert_image_color(image, [keep_index]*3, [0, 0, 0], inverse=True)
			image = convert_image_color(image, [keep_index]*3, [255, 255, 255])
			name = os.path.splitext(filename)[0]
			frame_dict[name] = image
		logger.info("Got N frames for mask: %d" % len(frame_dict))
		return frame_dict

	def _get_frames(self, h5, nframes=10000, mode="BGR"):
		"""
		Retrieve the color or depth frames
		"""
		assert mode in ["BGR", "Z"], "Frames are with mode=BGR(color) or Z(depth)"
		frames_dict = {}
		try:
			for key in h5:
				if "FRAME" not in key: continue
				if nframes <= 0: break
				frame = np.array(h5[key]["RAW"][self.camera][mode])
				# convert color to RGB
				if mode == "BGR":
					frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frames_dict[key] = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
				nframes -= 1
			logger.info("Total number of %s frames: %d" % (mode, len(frames_dict)))
		except Exception:
			logger.error("Cannot get frames for mode=%s" % mode)
		return frames_dict
	
	def _get_camera_parameters(self, h5info):
		"""
		setting the camera parameters to be used in depth transformation.
		"""
		try:
			self.color_dimension = np.array(h5info[self.camera]["COLOR_DIMENSION"])
			logger.info("Color dimension (height, width): %s" % str(self.color_dimension))
			self.color_rotation = np.array(h5info[self.camera]["COLOR_ROTATION"])
			logger.info("Color rotation: %s" % str(self.color_rotation))
			self.color_translation = np.array(h5info[self.camera]["COLOR_TRANSLATION"])
			logger.info("Color translation: %s" % str(self.color_translation))
			# intrinsics with fx, fy, cx, cy
			self.color_intrinsics = np.array(h5info[self.camera]["INTRINSICS_COLOR"])
			logger.info("Color intrinsics (fx, fy, cx, cy): %s" % str(self.color_intrinsics))

			self.depth_dimension = np.array(h5info[self.camera]["DEPTH_DIMENSION"])
			self.depth_intrinsics = np.array(h5info[self.camera]["INTRINSICS_DEPTH"])
			self.fps = np.array(h5info[self.camera]["FPS"])[0]
			logger.info("Video FPS: %d" % self.fps)

			# use height and width from color as default for all
			self.frame_height, self.frame_width = self.color_dimension[0], self.color_dimension[1]
		except Exception:
			logger.error("Not all camera infor are found!! Existing keys are:")
			for key in h5info[self.camera]:
				logger.info("key: %s" % key)
