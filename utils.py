import numpy as np
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_image_color(img, rgb_original, rgb_converted, inverse=False, src_img=None):
	"""
	Function to convert original color pixels into other colors.

	Parameters:
		img: is the image (numpy.array) with shape: (w, h, 3)
		rgb_original_list: a list of 1D array with 3 values, e.g.
			[[128, 0, 128], [0, 64, 128]] pixels with these colors will be
			replaced.
		rgb_converted: 1D array with 3 values, e.g. [255, 255, 255], the pixels
			above are replaced with this color.
		inverse: bool, if True, then pixels WITHOUT rgb_original is
			replaced with rgb_converted
	"""
	assert len(rgb_converted) == 3, "rgb length of 3"
	assert len(rgb_original) == 3, "rgb length of 3"
	data = np.array(img)
	src_img = data if src_img is None else src_img
	# red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
	red, green, blue = src_img[:, :, 0], src_img[:, :, 1], src_img[:, :, 2]
	# Original value
	r1, g1, b1 = rgb_original[0], rgb_original[1], rgb_original[2]
	if inverse:
		mask = (red != r1) | (green != g1) | (blue != b1)
	else:
		mask = (red == r1) & (green == g1) & (blue == b1)
	# Value that we want to replace it with
	r2, g2, b2 = rgb_converted[0], rgb_converted[1], rgb_converted[2]
	data[:, :, :3][mask] = [r2, g2, b2]
	return data


def get_boundary(masks_list, enlarge=0.1):
	"""
	Given the masks in the masks list, return the bounding box, xmin,ymin to xmax,ymax.
	Parameters:
		masks_list: (np.array list), masks are represented with np.array
		enlarge: (float) in [0., 1.], adding edge to the bounding box by xmin -= 0.1 * xlen
			because the persons are standing and in general xlen << ylen, a factor of two
			is applied to y (width) ymin -= 0.1 * ylen * 2

	Returns:
		tuple of (xmin, ymin, xmax, ymax), where x is height, y is width
	"""
	assert 0 <= enlarge < 1.0, "parameter enlarge: %.2f must in [0, 1)" % enlarge
	assert len(masks_list) and len(
		masks_list[0].shape) >= 2, "Masks must not be empty"
	npix_x, npix_y = masks_list[0].shape[0:2]
	logger.info("Frames initial shape: (%d, %d)" % (npix_x, npix_y))
	bounds = [99999, 0, 99999, 0]
	for mask in masks_list:
		# indices with non-zero values in mask
		if len(mask.shape) == 3:
			# mask is a 3D image
			mask = mask[:, :, 0]
		# print("mask", np.nonzero(mask))
		x_idx, y_idx = np.nonzero(mask)
		bounds = [
			min(bounds[0], np.percentile(x_idx, 1)),
			max(bounds[1], np.percentile(x_idx, 99)),
			min(bounds[2], np.percentile(y_idx, 1)),
			max(bounds[3], np.percentile(y_idx, 99))
		]
	xlen, ylen = bounds[1] - bounds[0], bounds[3] - bounds[2]
	assert xlen > 0 and ylen > 0, "length in x and y must be positive"
	bounds[0] = max(int(bounds[0] - xlen * enlarge), 0)
	bounds[1] = min(int(bounds[1] + xlen * enlarge), npix_x)
	bounds[2] = max(int(bounds[2] - ylen * enlarge * 2), 0)
	bounds[3] = min(int(bounds[3] + ylen * enlarge * 2), npix_y)
	logger.info("Frames boundary X: (%d, %d), Y: (%d, %d)" %
				(bounds[0], bounds[1], bounds[2], bounds[3]))
	xlen, ylen = bounds[1] - bounds[0], bounds[3] - bounds[2]
	logger.info("Cropped box X (height): %d, Y (width): %d" % (xlen, ylen))
	return bounds

def pcd_clear_ground(pcd):
	"""
	Ground is defined around the minimum on Y axis.
	"""
	pcdarr = np.asarray(pcd.points)
	pcdcol = np.asarray(pcd.colors)
	# use color to select the points, if the points are having
	# the same value for color in x, y, z, e.g. values: [0.49019608 0.49019608 0.49019608]
	# then remove them
	idx_select = np.array([i for i, v in enumerate(pcdcol) if v[0]==v[1]==v[2]])
	logger.info("Initial point cloud: %s, center: %s" % (str(pcd), str(pcd.get_center())))
	pcd = pcd.select_by_index(idx_select, invert = True)
	logger.info("Point cloud ground cleared: %s, center: %s" % (str(pcd), str(pcd.get_center())))
	return pcd

def pcd_outlier_removal(pcd, mode="statistical", config={}):
	assert mode in ["statistical", "radius"], "Mode: %s not found" % mode
	logger.info("Point cloud: %s, center: %s" % (str(pcd), str(pcd.get_center())))
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
	logger.info("Point cloud after removal: %s, center: %s" % (str(pcd), str(pcd.get_center())))
	return pcd
