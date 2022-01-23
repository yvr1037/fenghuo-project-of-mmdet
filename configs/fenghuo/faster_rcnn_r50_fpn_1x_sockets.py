"""配置configs，定义模型结构,
该配置文件放到指定位置"""

# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
        # type = 'FasterRCNN',
        roi_head=dict(
            bbox_head=dict(num_classes=7),
            mask_head=dict(
                            #  type='FCNMaskHead', 
                            #  num_convs=0, 
                            #  in_channels=2048, 
                            #  conv_out_channels=256, 
                        #    num_classes=7, 
                        #    loss_mask=dict( type='CrossEntropyLoss', 
                        #               use_mask=True, loss_weight=1.0 )
                        )
         )
    )

# Modify dataset related settings
dataset_type = 'CocoDataset' 
classes = ('out1','out2','out3','in1','in2','in3','2',) # dataset processing and find categories

data = dict(
    train=dict(
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/train/',
        classes=classes,
        ann_file='/home/gyf/projects/mmdetection/data/sockets/coco_train.json'),
    val=dict(
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/val/',
        classes=classes,
        ann_file='/home/gyf/projects/mmdetection/data/sockets/coco_val.json'),
    test=dict(
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/val/',
        classes=classes,
        ann_file='/home/gyf/projects/mmdetection/data/sockets/coco_val.json'))

# We can use the pre-trained Faster RCNN model to obtain higher performance
load_from = '/home/gyf/projects/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


