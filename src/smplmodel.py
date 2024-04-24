import argparse
import cv2
from time import time
from datetime import timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import logging
import torch
import smplx

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class SMPLModel:
	"""
	Based on repository: https://github.com/vchoutas/smplx	
	"""

	# pose parameters found:
	# https://github.com/mkocabas/body-model-visualizer/blob/3c74885ae1ebddf26db31dd95763d6e486e3d8c1/main.py#L264
	POSE_PARAMS = {
            'SMPL': {
                'body_pose': torch.zeros(1, 23, 3),
                'global_orient': torch.zeros(1, 1, 3),
            },
            'SMPLX': {
                'body_pose': torch.zeros(1, 21, 3),
                'global_orient': torch.zeros(1, 1, 3),
                ### 'left_hand_pose': torch.zeros(1, 15, 3),
                ### 'right_hand_pose': torch.zeros(1, 15, 3),
                # 'jaw_pose': torch.zeros(1, 1, 3),
                # 'leye_pose': torch.zeros(1, 1, 3),
                # 'reye_pose': torch.zeros(1, 1, 3),
            },
            'MANO': {
                'hand_pose': torch.zeros(1, 15, 3),
                'global_orient': torch.zeros(1, 1, 3),
            },
            'FLAME': {
                'global_orient': torch.zeros(1, 1, 3),
                'jaw_pose': torch.zeros(1, 1, 3),
                'neck_pose': torch.zeros(1, 1, 3),
                'leye_pose': torch.zeros(1, 1, 3),
                'reye_pose': torch.zeros(1, 1, 3),
            },
	}
	def __init__(self, config={}):
		"""
		Initialization.
		
		Parameters:
			filenames, list of str, full path to the filnames of the input point
				clouds
		"""
		assert "model_folder" in config, "Need model folder to setup SMPL/X/H/MANO/FLAME model"
		model_folder = config["model_folder"]
		model_ext = config.get("model_ext", "npz")
		gender = config.get("gender", "neutral")
		use_face_contour = config.get("use_face_contour", True)
		num_betas = config.get("num_betas", 10)
		num_expression_coeffs = config.get("num_expression_coeffs", 10)
		self.model_type = config.get("model_type", "smplx")
		self.model = smplx.create(model_folder, model_type=self.model_type,
						 gender=gender, use_face_contour=use_face_contour,
						 num_betas=num_betas,
						 num_expression_coeffs=num_expression_coeffs,
						 ext=model_ext)
		logger.info("Loaded model %s from folder: %s" % (self.model_type, model_folder))
		logger.info("%s" % str(self.model))

	def pose_parameters(self, pose_dict={}, use_random=True):
		# https://github.com/mkocabas/body-model-visualizer/blob/3c74885ae1ebddf26db31dd95763d6e486e3d8c1/main.py  # L1454
		para = self.POSE_PARAMS[self.model_type.upper()]
		for key in para:
			if key in pose_dict:
				para[key] = pose_dict[key]
			elif use_random:
				para[key] = torch.randn(para[key].shape, dtype=torch.float32)
		logger.info("Updated pose parameters: %s" % str(para))
		return para

	def generate_model(self):
		betas = torch.randn([1, self.model.num_betas], dtype=torch.float32)
		logger.info("The betas parameters: %s" % str(betas))
		expression = torch.randn([1, self.model.num_expression_coeffs], dtype=torch.float32)
		logger.info("The expression parameters: %s" % str(expression))
		# passing all pose parameters with **pose_para, equally one can do one by one with
		# body_pose=pose_para["body_pose"], ...
		# check: https://github.com/vchoutas/smplx/blob/master/smplx/body_models.py
		bdpose = torch.zeros(1, 21, 3)
		bdpose[0,1] = torch.tensor([0., 0.5, 0.])
		bdpose[0,5] = torch.tensor([0., 1.0, 0.])
		bdpose[0,15] = torch.tensor([0., -1.0, 0.5])
		pose_para = self.pose_parameters({"body_pose": bdpose})
		# print("Body Pose", pose_para["body_pose"])
		output = self.model(betas=betas, expression=expression,
					   return_verts=True, **pose_para)
					   # return_verts=True, body_pose=pose_para["body_pose"])
		self.vertices = output.vertices.detach().cpu().numpy().squeeze()
		logger.info("Vertices shape: %s" % str(self.vertices.shape))
		self.joints = output.joints.detach().cpu().numpy().squeeze()
		logger.info("Joints shape: %s" % str(self.joints.shape))
		logger.info("Joints values: %s" % str(self.joints))

	def vertice2pcd(self, outname):
		# convert generated model vertices to point clouds
		assert hasattr(self, "vertices"), "Vertices are not found for current model"
		points = o3d.utility.Vector3dVector(self.vertices)
		pcd = o3d.geometry.PointCloud(points=points)
		o3d.io.write_point_cloud(outname, pcd)
		logger.info("Point clouds saved to: %s" % outname)

	# fit smpl parameters to point clouds?
	# https://github.com/ZhengZerong/im2smpl/blob/master/fit_3d_accurate.py

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		# ~/Workarea/SFLScientific/Verizon_3Dface/analysis/
		# smpl_viz/body-model-visualizer/data/body_models/
		"--model-folder", "-m", default=None, type=str, required=True,
		help="Your input smpl model folder.",
	)
	parser.add_argument(
		"--output", "-o", default="outputs/smpl_model", type=str, required=False,
		help="Your output folder.",
	)

	args = parser.parse_args()
	os.makedirs(args.output, exist_ok=True)

	sm = SMPLModel({"model_folder": args.model_folder})
	sm.generate_model()
	sm.vertice2pcd(os.path.join(args.output, "vertices.ply"))

	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	"""
	Execute example:
		# in data/body_models, it expects folder of smplx or smpl, etc
		python3 smplmodel.py -m data/body_models
	"""
	main()
