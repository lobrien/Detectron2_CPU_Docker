import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

im = cv2.imread("./backlit.jpg")
print(f"Image shape {im.shape}")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
print(f"The mask is {outputs['instances'].pred_masks}")
#print(f"The keypoints are {outputs['instances'].pred_keypoints}")

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
v2 = Visualizer(np.empty_like(im[:,:,::-1])) 
out = v.draw_instance_predictions( outputs['instances'].to("cpu"))
cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])

mask = outputs["instances"].pred_masks.cpu().detach().numpy()
# TODO: This should be mask for shape in mask.shape and then multiple cropped images 
# TODO: There's some way to ignore features that are smaller than, say, 50x50 pixels
mask = mask.reshape(mask.shape[1:])
bmask = v2.draw_binary_mask(mask)
mask_img = bmask.get_image()[:,:,::-1]
white = np.copy(mask_img)
for row in range(len(white)) : 
    for col in range(len(white[row])):
        if white[row][col][1] > 0 : 
            white[row][col] = [255,255,255]
res = cv2.bitwise_and(im,white)
cv2.imwrite('cropped.jpg', res)
