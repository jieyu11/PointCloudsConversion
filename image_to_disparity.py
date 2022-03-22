import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import open3d as o3d
import logging
import argparse
from time import time
from datetime import timedelta
from utils import convert_image_color, get_boundary

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class stageH5conversion:
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
		# self._clean_depth_with_mask()
	
	def create_point_clouds(self, outdir=None):
		"""
		Create point clouds from RGBD images from create_rgbd_images() function.
		"""
		if not hasattr(self, 'frames_rgbd'):
			logger.error("Must have RGBD images to create point clouds.")
			return
		if outdir is not None:
			os.makedirs(outdir, exist_ok=True)
		# https://stackoverflow.com/questions/62809091/update-camera-intrinsic-parameter-in-open3d-python
		# Example CAM0 parameters: fx, fy, cx, cy = 930.20984 930.0072  636.0003  359.95996
		# Intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
		# cam.intrinsic_matrix =  [[930.2, 0.00, 636.0] , [0.00, 930.0, 359.9], [0.00, 0.00, 1.00]]
		# default options:
		# 	o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault
		# 	or PrimeSenseDefault ?
		cam = o3d.camera.PinholeCameraIntrinsic(
			width=self.frame_width, height=self.frame_height,
			fx=self.color_intrinsics[0], fy=self.color_intrinsics[1],
			cx=self.color_intrinsics[2], cy=self.color_intrinsics[3]
			)
		self.point_clouds = {}
		for name, frame_rgbd in self.frames_rgbd.items():
			pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			    frame_rgbd, cam)

			# Flip it, otherwise the pointcloud will be upside down
			pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
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
			self.frames_rgbd[name] = o3d.geometry.RGBDImage.create_from_color_and_depth(
			    o3d.geometry.Image(frame_color.astype(np.uint8)),
                o3d.geometry.Image(frame_depth.astype(np.float32))
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
				# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frames_dict[key] = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
				nframes -= 1
			logger.info("Total number of %s frames: %d" % (mode, len(frames_dict)))
		except Exception:
			logger.error("Cannot get frames for mode=%s" % mode)
		return frames_dict
	
def convert_left_right_to_disparity(image_left, image_right, output):
	"""
	setting the camera parameters to be used in depth transformation.
	"""
	imleft = cv2.imread(image_left)
	imright = cv2.imread(image_right)
	stereo = cv2.StereoCamera()
	disp = stereo.findDisparityMap()

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--image-left", "-l", default=None, type=str, required=True,
		help="Left image file.",
	)
	parser.add_argument(
		"--image-right", "-r", default=None, type=str, required=True,
		help="Right image file.",
	)
	parser.add_argument(
		"--output", "-o", default="outputs", type=str, required=False,
		help="Your output directory",
	)

	args = parser.parse_args()
	logger.info("Reading h5 file: %s" % args.h5)
	sc = stageH5conversion(args.h5, config={
		"camera": args.camera,
		"nframes": args.nframes,
		"mask": args.mask
		})
	# sc.draw_color(os.path.join(args.output, "color"), args.ndraw)
	# sc.draw_depth(os.path.join(args.output, "depth"), args.ndraw)
	for mode in ["color", "depth"]:
		# do mask images before cropping
		sc.mask_images(mode=mode)
		sc.crop_images(mode=mode)
	sc.create_rgbd_images()
	# sc.create_point_clouds(outdir=os.path.join(args.output, "point_clouds"))
	sc.draw_rgbd(os.path.join(args.output, "rgbd"), args.ndraw)

	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	main()
