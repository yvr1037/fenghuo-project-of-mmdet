import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector
import torch

# device = torch.device("cpu")

# os.environ["CUDA_VISIBLE_DEVICES"]=""
# import pdb; pdb.set_trace()
config_file = '../configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py'
checkpoint_file = '../work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets/epoch_17.pth'
model = init_detector(config_file, checkpoint_file)
# img_dir = '../data/sockets/train/'
# out_dir = 'results/'
img = '00019.jpg'
# img = mmcv.imread(img)
result = inference_detector(model, img)
model.show_result(img, result, out_file='testOut6.jpg')
# model.show_result(img, result, model.CLASSES)

print(result)