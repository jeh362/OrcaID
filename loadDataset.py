
import os

from roboflow import Roboflow



rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("jacqueline-eb2ts").project("orca-dnre4")
version = project.version(2)
dataset = version.download("coco-segmentation")