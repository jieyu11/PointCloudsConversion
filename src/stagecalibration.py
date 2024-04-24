import os
import logging
import json

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)

class StageCalibration:
	def __init__(self, filename):
		"""
		Initialization.
		
		Parameters:
			filename, str, full path to the filname of the stage calibration
				content of this can be found in Example_CalibratedStage.json
				as an example.
		"""
		assert os.path.exists(filename), "File %s not exist" % filename
		with open(filename) as f:
			self.data = json.load(f)
		assert "rigs" in self.data, "Key: 'rigs' is not found in %s" % filename

	def get_config(self):
		"""
		Get the config for all rigs
		"""
		logger.info("Found # of rigs: %d" % len(self.data["rigs"]))
		config = {}
		for rig in self.data["rigs"]:
			config["rig%d" % rig["id"]] = {
				"baseline": rig["baseline"],
				"extrinsics": rig["homogeneousExtrinsics"],
				"intrinsics": {
					"left": rig["leftCamera"]["intrinsics"],
					"right": rig["rightCamera"]["intrinsics"],
				},
				"focallength": {
					"left": rig["leftCamera"]["focalLength"], 
					"right": rig["rightCamera"]["focalLength"], 
				},
				"width": {
					"left": rig["leftCamera"]["width"],
					"right": rig["rightCamera"]["width"],
				},
				"height": {
					"left": rig["leftCamera"]["height"],
					"right": rig["rightCamera"]["height"],
				},
			}
		for key, rig in config.items():
			logger.info("Config for %s" % key)
			logger.info("%s" % str(rig))
		return config	
