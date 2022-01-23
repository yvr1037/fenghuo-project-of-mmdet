"""本模块用于放置configs，定义模型结构,
该配置文件将放到`/home/gyf/projects/mmdetection/configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py"""

# The new config inherits a base config to highlight the necessary modification

_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=8),
        mask_head=dict(num_classes=8)
        )
    )

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('out1','out2','out3','in1','in2','in3','2','')  # dataset processing and find categories
data = dict(
    train=dict(
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/train/',
        classes=classes,
        ann_file='/home/gyf/projects/mmdetection/data/sockets/train/annotation_coco.json'),
    val=dict(
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/train/',
        classes=classes,
        ann_file='/home/gyf/projects/mmdetection/data/sockets/train/annotation_coco.json'),
    test=dict(
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/train/',
        classes=classes,
        ann_file='/home/gyf/projects/mmdetection/data/sockets/train/annotation_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = '/home/gyf/projects/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'