数据处理的思路参考了朋友逯润雨的思路,有很大帮助！！
逯同学blog:https://lry89757.github.io/2021/11/09/mmdet-project-of-fenghuo/

#### Perface

> https://zhuanlan.zhihu.com/p/256344471
> https://zhuanlan.zhihu.com/p/337375549
> https://zhuanlan.zhihu.com/p/431215846
> https://aistudio.baidu.com/paddle/forum/topic/show/990118

​	首先了解一下[coco数据集的制作](http://cocodataset.org)，MS COCO的全称是Microsoft Common Objects in Context，起源于微软于2014年出资标注的Microsoft COCO数据集。COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene understanding为目标，主要从复杂的日常场景中截取，图像中的目标通过精确的segmentation进行位置的标定。图像包括91类目标，328,000影像和2,500,000个label。数据集主要解决3个问题：目标检测，目标之间的上下文关系，目标的2维上的精确定位.

> COCO一共有5种不同任务分类，分别是目标检测、关键点检测、语义分割、场景分割和图像描述。COCO数据集的标注文件以JSON格式保存，官方的注释文件有仨 captions_type.json instances_type.json person_keypoints_type.json，其中的type是 train/val/test+year

​ sockets检测任务我参照一些博客上：

> [VOC和COCO数据集制作](https://aistudio.baidu.com/paddle/forum/topic/show/990118)

写的转换后的json文件中annonation里面的**‘bbox’**确实转换的时候出现了很大的问题，第一次是全部是0，第二次我没有直接转换而是直接按照labelme rectangle标注格式的坐标计算推理出框的相关信息但是最后训练还是出现了大问题,写脚本转coco格式这条路目前在我这里是失败了，最重要做检测的数据annotation[‘bbox’]数据信息没有转成功真是致命的：

![LbtVF2.png](https://s6.jpg.cm/2021/12/23/LbtVF2.png)
	

​	之前做的图片处理可能是因为我还没有学习很多目标检测方面的东西以及工程能力太弱，不算一个小项目只是一个小任务我就做了好几天最后也没用上，其实还是感觉沟通不及时不知道最后要达到什么效果，直到最后才明白是为了后续组里无论是做检测还是分割任务时便于提高精度达到检测的更好的效果.

#### mmdet.api的探讨

> 2021.12.20 晚 709

连续搞了几天sockets数据集的处理都没整明白，好菜啊，

明确任务:做sockets图片插孔的目标检测和分类任务：

先看[mmdetection](https://mmdetection.readthedocs.io/en/latest/api.html)介绍:

<u>感谢🦌同学</u>🐂

源码追究一下 `mmdet.apis.inference_detector()`,这里大部分有助于快速完成项目参考了[逯同学](https://lry89757.github.io/2021/12/05/feng-huo-xiang-mu-fen-lei-ren-wu-zong-jie/#toc-heading-3)这一部分，后面还是要花时间读一下mmdetection官方指导文档.

dataloader of mmdetection

蒋哥一直推荐不要每次做项目都是转换成coco/voc格式，这样比较麻烦的原因在于每次labelme标注的信息文件格式很有可能不一样，每次转化都要重写脚本是非常痛苦的(但是我觉得前几次还是有必要转格式积累经验的)，当然现阶段我也不会写dataloader转化，只能硬写转化脚本,:angry:好气呀！

看了组里其他同学的博客(强烈安利[逯同学的博客](https://lry89757.github.io/2021/12/05/feng-huo-xiang-mu-fen-lei-ren-wu-zong-jie/)明白MMDetection需要深入理解Rigistry机制的，需要参照之前溯源的过程去寻找。

##### sockets插孔数据再处理:thinking:



#### sockets检测

> 2021.12.21晚 709

**gyf@ubuntu ~/projects/mmdetection**

`% python tools/train.py configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py`

##### 2021-12-21 21:51:09,265 - mmdet - INFO - Environment info:

sys.platform: linux
Python: 3.9.7 (default, Sep 16 2021, 13:09:58) [GCC 7.5.0]
CUDA available: True
GPU 0,1,2,3: GeForce RTX 2080 Ti
CUDA_HOME: /usr/local/cuda
NVCC: Build cuda_11.0_bu.TC445_37.28845127_0
GCC: gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
PyTorch: 1.10.1
PyTorch compiling details: PyTorch built with:

  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.3
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.2
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON,

TorchVision: 0.11.2
OpenCV: 4.5.4
MMCV: 1.4.0
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.3

##### MMDetection: 2.18.0+6cf9aa1

2021-12-21 21:51:10,501 - mmdet - INFO - Distributed training: False
2021-12-21 21:51:11,750 - mmdet - INFO - Config:
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'COCODataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/gyf/projects/mmdetection/data/sockets/train/annotation_coco.json',
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                           (1333, 768), (1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        classes=('out1', 'out2', 'out3', 'in1', 'in2', 'in3', '2', '')),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/gyf/projects/mmdetection/data/sockets/val/annotation_coco.json',
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('out1', 'out2', 'out3', 'in1', 'in2', 'in3', '2', '')),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/gyf/projects/mmdetection/data/sockets/val/annotation_coco.json',
        img_prefix='/home/gyf/projects/mmdetection/data/sockets/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('out1', 'out2', 'out3', 'in1', 'in2', 'in3', '2', '')))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/gyf/projects/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
resume_from = None
workflow = [('train', 1)]
classes = ('out1', 'out2', 'out3', 'in1', 'in2', 'in3', '2', '')
work_dir = './work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets'
gpu_ids = range(0, 1)

2021-12-21 21:51:11,751 - mmdet - INFO - Set random seed to 1229834670, deterministic: False
2021-12-21 21:51:12,198 - mmdet - INFO - initialize ResNet with init_cfg {'type': 'Pretrained', 'checkpoint': 'open-mmlab://detectron2/resnet50_caffe'}
2021-12-21 21:51:12,198 - mmcv - INFO - load model from: open-mmlab://detectron2/resnet50_caffe
2021-12-21 21:51:12,199 - mmcv - INFO - load checkpoint from openmmlab path: open-mmlab://detectron2/resnet50_caffe
2021-12-21 21:51:12,307 - mmcv - WARNING - The model and loaded state dict do not match exactly

unexpected key in source state_dict: conv1.bias

2021-12-21 21:51:12,340 - mmdet - INFO - initialize FPN with init_cfg {'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
2021-12-21 21:51:12,383 - mmdet - INFO - initialize RPNHead with init_cfg {'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01}
2021-12-21 21:51:12,391 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
2021-12-21 21:51:15,458 - mmdet - INFO - load checkpoint from local path: /home/gyf/projects/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
2021-12-21 21:51:15,617 - mmdet - WARNING - The model and loaded state dict do not match exactly

size mismatch for roi_head.bbox_head.fc_cls.weight: copying a param with shape torch.Size([81, 1024]) from checkpoint, the shape in current model is torch.Size([9, 1024]).
size mismatch for roi_head.bbox_head.fc_cls.bias: copying a param with shape torch.Size([81]) from checkpoint, the shape in current model is torch.Size([9]).
size mismatch for roi_head.bbox_head.fc_reg.weight: copying a param with shape torch.Size([320, 1024]) from checkpoint, the shape in current model is torch.Size([32, 1024]).
size mismatch for roi_head.bbox_head.fc_reg.bias: copying a param with shape torch.Size([320]) from checkpoint, the shape in current model is torch.Size([32]).
size mismatch for roi_head.mask_head.conv_logits.weight: copying a param with shape torch.Size([80, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([8, 256, 1, 1]).
size mismatch for roi_head.mask_head.conv_logits.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([8]).
2021-12-21 21:51:15,623 - mmdet - INFO - Start running, host: gyf@ubuntu, work_dir: /mnt/data01/home/gyf/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets
2021-12-21 21:51:15,623 - mmdet - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) StepLrUpdaterHook
(NORMAL      ) CheckpointHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook

----

before_train_epoch:
(VERY_HIGH   ) StepLrUpdaterHook
(NORMAL      ) NumClassCheckHook
(LOW         ) IterTimerHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook

before_train_iter:
(VERY_HIGH   ) StepLrUpdaterHook
(LOW         ) IterTimerHook
(LOW         ) EvalHook

---

after_train_iter:
(ABOVE_NORMAL) OptimizerHook
(NORMAL      ) CheckpointHook
(LOW         ) IterTimerHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook

---

after_train_epoch:
(NORMAL      ) CheckpointHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook

---

before_val_epoch:
(NORMAL      ) NumClassCheckHook
(LOW         ) IterTimerHook
(VERY_LOW    ) TextLoggerHook

---

before_val_iter:
(LOW         ) IterTimerHook

---

after_val_iter:
(LOW         ) IterTimerHook

---

after_val_epoch:
(VERY_LOW    ) TextLoggerHook

---

after_run:
(VERY_LOW    ) TextLoggerHook

---

**epoch：1**

```pyhon
2021-12-21 21:51:15,623 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
2021-12-21 21:51:15,623 - mmdet - INFO - Checkpoints will be saved to /mnt/data01/home/gyf/projects/mmdetection/work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets by HardDiskBackend.
2021-12-21 21:51:30,877 - mmdet - INFO - Epoch [1][50/930]      lr: 1.978e-03, eta: 0:56:09, time: 0.303, data_time: 0.051, memory: 3050, loss_rpn_cls: 0.0593, loss_rpn_bbox: 0.0060, loss_cls: 0.4836, acc: 88.0762, loss_bbox: 0.0283, loss_mask: 0.5316, loss: 1.1088
2021-12-21 21:51:42,944 - mmdet - INFO - Epoch [1][100/930]     lr: 3.976e-03, eta: 0:50:09, time: 0.241, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0324, loss_rpn_bbox: 0.0057, loss_cls: 0.1005, acc: 98.5547, loss_bbox: 0.0447, loss_mask: 0.3602, loss: 0.5434
2021-12-21 21:51:54,878 - mmdet - INFO - Epoch [1][150/930]     lr: 5.974e-03, eta: 0:47:54, time: 0.239, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0336, loss_rpn_bbox: 0.0059, loss_cls: 0.0944, acc: 98.4277, loss_bbox: 0.0476, loss_mask: 0.3338, loss: 0.5154
2021-12-21 21:52:07,173 - mmdet - INFO - Epoch [1][200/930]     lr: 7.972e-03, eta: 0:46:59, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0321, loss_rpn_bbox: 0.0064, loss_cls: 0.0856, acc: 98.5020, loss_bbox: 0.0430, loss_mask: 0.3231, loss: 0.4902
2021-12-21 21:52:19,585 - mmdet - INFO - Epoch [1][250/930]     lr: 9.970e-03, eta: 0:46:27, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0250, loss_rpn_bbox: 0.0052, loss_cls: 0.1124, acc: 97.8770, loss_bbox: 0.0575, loss_mask: 0.3223, loss: 0.5224
2021-12-21 21:52:31,916 - mmdet - INFO - Epoch [1][300/930]     lr: 1.197e-02, eta: 0:45:58, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0263, loss_rpn_bbox: 0.0057, loss_cls: 0.1151, acc: 97.8320, loss_bbox: 0.0521, loss_mask: 0.2696, loss: 0.4687
2021-12-21 21:52:44,130 - mmdet - INFO - Epoch [1][350/930]     lr: 1.397e-02, eta: 0:45:30, time: 0.244, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0306, loss_rpn_bbox: 0.0071, loss_cls: 0.1030, acc: 98.2188, loss_bbox: 0.0496, loss_mask: 0.3332, loss: 0.5235
2021-12-21 21:52:56,471 - mmdet - INFO - Epoch [1][400/930]     lr: 1.596e-02, eta: 0:45:10, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0308, loss_rpn_bbox: 0.0069, loss_cls: 0.0996, acc: 98.3027, loss_bbox: 0.0423, loss_mask: 0.3790, loss: 0.5587
2021-12-21 21:53:09,135 - mmdet - INFO - Epoch [1][450/930]     lr: 1.796e-02, eta: 0:44:59, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0285, loss_rpn_bbox: 0.0061, loss_cls: 0.1043, acc: 98.0840, loss_bbox: 0.0431, loss_mask: 0.3195, loss: 0.5014
2021-12-21 21:53:21,833 - mmdet - INFO - Epoch [1][500/930]     lr: 1.996e-02, eta: 0:44:48, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0290, loss_rpn_bbox: 0.0080, loss_cls: 0.1086, acc: 97.9746, loss_bbox: 0.0502, loss_mask: 0.3295, loss: 0.5252
2021-12-21 21:53:34,545 - mmdet - INFO - Epoch [1][550/930]     lr: 2.000e-02, eta: 0:44:38, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0294, loss_rpn_bbox: 0.0069, loss_cls: 0.1129, acc: 97.9414, loss_bbox: 0.0471, loss_mask: 0.3062, loss: 0.5025
2021-12-21 21:53:46,741 - mmdet - INFO - Epoch [1][600/930]     lr: 2.000e-02, eta: 0:44:18, time: 0.244, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0271, loss_rpn_bbox: 0.0074, loss_cls: 0.1000, acc: 98.1562, loss_bbox: 0.0464, loss_mask: 0.3042, loss: 0.4850
2021-12-21 21:53:59,443 - mmdet - INFO - Epoch [1][650/930]     lr: 2.000e-02, eta: 0:44:07, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0268, loss_rpn_bbox: 0.0074, loss_cls: 0.1002, acc: 98.0684, loss_bbox: 0.0420, loss_mask: 0.2872, loss: 0.4636
2021-12-21 21:54:12,305 - mmdet - INFO - Epoch [1][700/930]     lr: 2.000e-02, eta: 0:43:58, time: 0.257, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0260, loss_rpn_bbox: 0.0057, loss_cls: 0.1047, acc: 97.9590, loss_bbox: 0.0429, loss_mask: 0.2588, loss: 0.4381
2021-12-21 21:54:24,763 - mmdet - INFO - Epoch [1][750/930]     lr: 2.000e-02, eta: 0:43:43, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0194, loss_rpn_bbox: 0.0061, loss_cls: 0.0993, acc: 97.9316, loss_bbox: 0.0438, loss_mask: 0.2568, loss: 0.4255
2021-12-21 21:54:37,189 - mmdet - INFO - Epoch [1][800/930]     lr: 2.000e-02, eta: 0:43:28, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0247, loss_rpn_bbox: 0.0069, loss_cls: 0.0980, acc: 98.0312, loss_bbox: 0.0462, loss_mask: 0.2838, loss: 0.4596
2021-12-21 21:54:49,453 - mmdet - INFO - Epoch [1][850/930]     lr: 2.000e-02, eta: 0:43:12, time: 0.245, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0281, loss_rpn_bbox: 0.0059, loss_cls: 0.1013, acc: 98.1250, loss_bbox: 0.0431, loss_mask: 0.2680, loss: 0.4463
2021-12-21 21:55:02,245 - mmdet - INFO - Epoch [1][900/930]     lr: 2.000e-02, eta: 0:43:02, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0243, loss_rpn_bbox: 0.0065, loss_cls: 0.1063, acc: 98.0195, loss_bbox: 0.0462, loss_mask: 0.2775, loss: 0.4607
2021-12-21 21:55:09,711 - mmdet - INFO - Saving checkpoint at 1 epochs

[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 10.9 task/s, elapsed: 70s, ETA:     0s2021-12-21 21:56:22,062 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.69s).
Accumulating evaluation results...
DONE (t=0.26s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.017
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.013
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.013
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.013
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.197
2021-12-21 21:56:23,045 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.70s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.27s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.017
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.050
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.013
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.013
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.197
2021-12-21 21:56:24,155 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 21:56:24,156 - mmdet - INFO - Epoch(val) [1][764]    bbox_mAP: 0.0110, bbox_mAP_50: 0.0170, bbox_mAP_75: 0.0130, bbox_mAP_s: 0.1500, bbox_mAP_m: 0.0130, bbox_mAP_l: 0.0130, bbox_mAP_copypaste: 0.011 0.017 0.013 0.150 0.013 0.013, segm_mAP: 0.0110, segm_mAP_50: 0.0170, segm_mAP_75: 0.0120, segm_mAP_s: 0.0500, segm_mAP_m: 0.0130, segm_mAP_l: 0.0130, segm_mAP_copypaste: 0.011 0.017 0.012 0.050 0.013 0.013
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch 2:**

```
2021-12-21 21:56:39,033 - mmdet - INFO - Epoch [2][50/930]      lr: 2.000e-02, eta: 0:41:46, time: 0.296, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0272, loss_rpn_bbox: 0.0066, loss_cls: 0.0903, acc: 98.1641, loss_bbox: 0.0372, loss_mask: 0.2521, loss: 0.4135
2021-12-21 21:56:51,349 - mmdet - INFO - Epoch [2][100/930]     lr: 2.000e-02, eta: 0:41:34, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0257, loss_rpn_bbox: 0.0061, loss_cls: 0.0883, acc: 98.2012, loss_bbox: 0.0378, loss_mask: 0.2791, loss: 0.4369
2021-12-21 21:57:03,682 - mmdet - INFO - Epoch [2][150/930]     lr: 2.000e-02, eta: 0:41:22, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0273, loss_rpn_bbox: 0.0063, loss_cls: 0.0983, acc: 98.0742, loss_bbox: 0.0396, loss_mask: 0.2770, loss: 0.4485
2021-12-21 21:57:16,379 - mmdet - INFO - Epoch [2][200/930]     lr: 2.000e-02, eta: 0:41:13, time: 0.254, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0150, loss_rpn_bbox: 0.0047, loss_cls: 0.1047, acc: 97.8418, loss_bbox: 0.0442, loss_mask: 0.2601, loss: 0.4287
2021-12-21 21:57:28,914 - mmdet - INFO - Epoch [2][250/930]     lr: 2.000e-02, eta: 0:41:02, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0230, loss_rpn_bbox: 0.0060, loss_cls: 0.1007, acc: 97.8574, loss_bbox: 0.0425, loss_mask: 0.2352, loss: 0.4074
2021-12-21 21:57:41,438 - mmdet - INFO - Epoch [2][300/930]     lr: 2.000e-02, eta: 0:40:51, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0236, loss_rpn_bbox: 0.0066, loss_cls: 0.0974, acc: 97.9766, loss_bbox: 0.0441, loss_mask: 0.2576, loss: 0.4293
2021-12-21 21:57:53,713 - mmdet - INFO - Epoch [2][350/930]     lr: 2.000e-02, eta: 0:40:38, time: 0.245, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0220, loss_rpn_bbox: 0.0067, loss_cls: 0.0966, acc: 97.9609, loss_bbox: 0.0380, loss_mask: 0.2470, loss: 0.4102
2021-12-21 21:58:06,207 - mmdet - INFO - Epoch [2][400/930]     lr: 2.000e-02, eta: 0:40:27, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0248, loss_rpn_bbox: 0.0070, loss_cls: 0.0962, acc: 98.0918, loss_bbox: 0.0447, loss_mask: 0.2851, loss: 0.4578
2021-12-21 21:58:18,727 - mmdet - INFO - Epoch [2][450/930]     lr: 2.000e-02, eta: 0:40:16, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0262, loss_rpn_bbox: 0.0078, loss_cls: 0.1020, acc: 97.9570, loss_bbox: 0.0420, loss_mask: 0.2593, loss: 0.4374
2021-12-21 21:58:31,463 - mmdet - INFO - Epoch [2][500/930]     lr: 2.000e-02, eta: 0:40:06, time: 0.255, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0200, loss_rpn_bbox: 0.0056, loss_cls: 0.1004, acc: 97.9414, loss_bbox: 0.0438, loss_mask: 0.2607, loss: 0.4305
2021-12-21 21:58:44,207 - mmdet - INFO - Epoch [2][550/930]     lr: 2.000e-02, eta: 0:39:57, time: 0.255, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0206, loss_rpn_bbox: 0.0061, loss_cls: 0.0948, acc: 97.9629, loss_bbox: 0.0432, loss_mask: 0.2799, loss: 0.4445
2021-12-21 21:58:56,856 - mmdet - INFO - Epoch [2][600/930]     lr: 2.000e-02, eta: 0:39:46, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0226, loss_rpn_bbox: 0.0078, loss_cls: 0.1000, acc: 97.8359, loss_bbox: 0.0479, loss_mask: 0.2761, loss: 0.4544
2021-12-21 21:59:09,542 - mmdet - INFO - Epoch [2][650/930]     lr: 2.000e-02, eta: 0:39:35, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0218, loss_rpn_bbox: 0.0067, loss_cls: 0.1068, acc: 97.7031, loss_bbox: 0.0465, loss_mask: 0.2699, loss: 0.4517
2021-12-21 21:59:22,348 - mmdet - INFO - Epoch [2][700/930]     lr: 2.000e-02, eta: 0:39:25, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0189, loss_rpn_bbox: 0.0057, loss_cls: 0.1085, acc: 97.7012, loss_bbox: 0.0498, loss_mask: 0.2500, loss: 0.4329
2021-12-21 21:59:34,816 - mmdet - INFO - Epoch [2][750/930]     lr: 2.000e-02, eta: 0:39:13, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0232, loss_rpn_bbox: 0.0063, loss_cls: 0.0955, acc: 97.9648, loss_bbox: 0.0437, loss_mask: 0.2768, loss: 0.4454
2021-12-21 21:59:47,408 - mmdet - INFO - Epoch [2][800/930]     lr: 2.000e-02, eta: 0:39:02, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0229, loss_rpn_bbox: 0.0062, loss_cls: 0.0971, acc: 97.8750, loss_bbox: 0.0437, loss_mask: 0.2533, loss: 0.4232
2021-12-21 21:59:59,688 - mmdet - INFO - Epoch [2][850/930]     lr: 2.000e-02, eta: 0:38:49, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0183, loss_rpn_bbox: 0.0052, loss_cls: 0.0937, acc: 97.9336, loss_bbox: 0.0432, loss_mask: 0.2523, loss: 0.4127
2021-12-21 22:00:12,573 - mmdet - INFO - Epoch [2][900/930]     lr: 2.000e-02, eta: 0:38:38, time: 0.258, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0199, loss_rpn_bbox: 0.0059, loss_cls: 0.0975, acc: 97.9297, loss_bbox: 0.0428, loss_mask: 0.2603, loss: 0.4264
2021-12-21 22:00:20,323 - mmdet - INFO - Saving checkpoint at 2 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 9.4 task/s, elapsed: 81s, ETA:     0s2021-12-21 22:01:43,823 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.11s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.86s).
Accumulating evaluation results...
DONE (t=0.34s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.019
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.013
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.016
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.194
2021-12-21 22:01:45,162 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.11s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.92s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.35s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.019
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.014
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.015
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.013
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.204
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.262
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.192
2021-12-21 22:01:46,746 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:01:46,747 - mmdet - INFO - Epoch(val) [2][764]    bbox_mAP: 0.0120, bbox_mAP_50: 0.0190, bbox_mAP_75: 0.0130, bbox_mAP_s: 0.1500, bbox_mAP_m: 0.0160, bbox_mAP_l: 0.0120, bbox_mAP_copypaste: 0.012 0.019 0.013 0.150 0.016 0.012, segm_mAP: 0.0120, segm_mAP_50: 0.0190, segm_mAP_75: 0.0140, segm_mAP_s: 0.1750, segm_mAP_m: 0.0150, segm_mAP_l: 0.0130, segm_mAP_copypaste: 0.012 0.019 0.014 0.175 0.015 0.013
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch 3:**

```
2021-12-21 22:02:01,913 - mmdet - INFO - Epoch [3][50/930]      lr: 2.000e-02, eta: 0:37:55, time: 0.302, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0186, loss_rpn_bbox: 0.0054, loss_cls: 0.1044, acc: 97.7559, loss_bbox: 0.0450, loss_mask: 0.2723, loss: 0.4458
2021-12-21 22:02:14,295 - mmdet - INFO - Epoch [3][100/930]     lr: 2.000e-02, eta: 0:37:43, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0204, loss_rpn_bbox: 0.0063, loss_cls: 0.0861, acc: 98.1270, loss_bbox: 0.0395, loss_mask: 0.2726, loss: 0.4250
2021-12-21 22:02:26,634 - mmdet - INFO - Epoch [3][150/930]     lr: 2.000e-02, eta: 0:37:31, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0221, loss_rpn_bbox: 0.0065, loss_cls: 0.1039, acc: 97.6504, loss_bbox: 0.0467, loss_mask: 0.2384, loss: 0.4177
2021-12-21 22:02:39,182 - mmdet - INFO - Epoch [3][200/930]     lr: 2.000e-02, eta: 0:37:20, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0207, loss_rpn_bbox: 0.0058, loss_cls: 0.0923, acc: 97.8906, loss_bbox: 0.0387, loss_mask: 0.2276, loss: 0.3852
2021-12-21 22:02:51,689 - mmdet - INFO - Epoch [3][250/930]     lr: 2.000e-02, eta: 0:37:08, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0196, loss_rpn_bbox: 0.0055, loss_cls: 0.0865, acc: 97.9414, loss_bbox: 0.0354, loss_mask: 0.2145, loss: 0.3615
2021-12-21 22:03:04,108 - mmdet - INFO - Epoch [3][300/930]     lr: 2.000e-02, eta: 0:36:57, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0237, loss_rpn_bbox: 0.0052, loss_cls: 0.0947, acc: 97.9160, loss_bbox: 0.0409, loss_mask: 0.2321, loss: 0.3966
2021-12-21 22:03:16,446 - mmdet - INFO - Epoch [3][350/930]     lr: 2.000e-02, eta: 0:36:44, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0189, loss_rpn_bbox: 0.0056, loss_cls: 0.1081, acc: 97.5762, loss_bbox: 0.0484, loss_mask: 0.2318, loss: 0.4127
2021-12-21 22:03:28,923 - mmdet - INFO - Epoch [3][400/930]     lr: 2.000e-02, eta: 0:36:33, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0248, loss_rpn_bbox: 0.0056, loss_cls: 0.1074, acc: 97.6680, loss_bbox: 0.0509, loss_mask: 0.2603, loss: 0.4490
2021-12-21 22:03:41,569 - mmdet - INFO - Epoch [3][450/930]     lr: 2.000e-02, eta: 0:36:22, time: 0.253, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0211, loss_rpn_bbox: 0.0058, loss_cls: 0.0928, acc: 97.9746, loss_bbox: 0.0404, loss_mask: 0.2682, loss: 0.4283
2021-12-21 22:03:54,270 - mmdet - INFO - Epoch [3][500/930]     lr: 2.000e-02, eta: 0:36:11, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0180, loss_rpn_bbox: 0.0053, loss_cls: 0.1024, acc: 97.9609, loss_bbox: 0.0469, loss_mask: 0.2769, loss: 0.4495
2021-12-21 22:04:07,070 - mmdet - INFO - Epoch [3][550/930]     lr: 2.000e-02, eta: 0:36:00, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0230, loss_rpn_bbox: 0.0068, loss_cls: 0.1004, acc: 97.8984, loss_bbox: 0.0437, loss_mask: 0.2519, loss: 0.4259
2021-12-21 22:04:19,619 - mmdet - INFO - Epoch [3][600/930]     lr: 2.000e-02, eta: 0:35:48, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0226, loss_rpn_bbox: 0.0058, loss_cls: 0.1000, acc: 97.7598, loss_bbox: 0.0415, loss_mask: 0.2337, loss: 0.4036
2021-12-21 22:04:32,238 - mmdet - INFO - Epoch [3][650/930]     lr: 2.000e-02, eta: 0:35:37, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0202, loss_rpn_bbox: 0.0061, loss_cls: 0.1085, acc: 97.5664, loss_bbox: 0.0453, loss_mask: 0.2252, loss: 0.4052
2021-12-21 22:04:45,017 - mmdet - INFO - Epoch [3][700/930]     lr: 2.000e-02, eta: 0:35:26, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0219, loss_rpn_bbox: 0.0058, loss_cls: 0.1043, acc: 97.6836, loss_bbox: 0.0450, loss_mask: 0.2512, loss: 0.4283
2021-12-21 22:04:57,497 - mmdet - INFO - Epoch [3][750/930]     lr: 2.000e-02, eta: 0:35:14, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0217, loss_rpn_bbox: 0.0069, loss_cls: 0.1021, acc: 97.8457, loss_bbox: 0.0493, loss_mask: 0.2673, loss: 0.4473
2021-12-21 22:05:10,218 - mmdet - INFO - Epoch [3][800/930]     lr: 2.000e-02, eta: 0:35:03, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0188, loss_rpn_bbox: 0.0059, loss_cls: 0.0950, acc: 97.8027, loss_bbox: 0.0392, loss_mask: 0.2219, loss: 0.3808
2021-12-21 22:05:22,822 - mmdet - INFO - Epoch [3][850/930]     lr: 2.000e-02, eta: 0:34:51, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0224, loss_rpn_bbox: 0.0066, loss_cls: 0.0955, acc: 97.7891, loss_bbox: 0.0423, loss_mask: 0.2192, loss: 0.3860
2021-12-21 22:05:35,648 - mmdet - INFO - Epoch [3][900/930]     lr: 2.000e-02, eta: 0:34:40, time: 0.257, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0281, loss_rpn_bbox: 0.0079, loss_cls: 0.0920, acc: 97.9434, loss_bbox: 0.0442, loss_mask: 0.2602, loss: 0.4324
2021-12-21 22:05:43,204 - mmdet - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.8 task/s, elapsed: 87s, ETA:     0s2021-12-21 22:07:12,110 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.78s).
Accumulating evaluation results...
DONE (t=0.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.016
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.027
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.019
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.238
2021-12-21 22:07:13,405 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.92s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.027
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.020
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.021
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.226
2021-12-21 22:07:15,042 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:07:15,042 - mmdet - INFO - Epoch(val) [3][764]    bbox_mAP: 0.0160, bbox_mAP_50: 0.0270, bbox_mAP_75: 0.0180, bbox_mAP_s: 0.2000, bbox_mAP_m: 0.0210, bbox_mAP_l: 0.0190, bbox_mAP_copypaste: 0.016 0.027 0.018 0.200 0.021 0.019, segm_mAP: 0.0170, segm_mAP_50: 0.0270, segm_mAP_75: 0.0190, segm_mAP_s: 0.3040, segm_mAP_m: 0.0200, segm_mAP_l: 0.0210, segm_mAP_copypaste: 0.017 0.027 0.019 0.304 0.020 0.021
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch 4:**

```
2021-12-21 22:07:29,912 - mmdet - INFO - Epoch [4][50/930]      lr: 2.000e-02, eta: 0:34:06, time: 0.295, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0219, loss_rpn_bbox: 0.0059, loss_cls: 0.0995, acc: 97.8066, loss_bbox: 0.0444, loss_mask: 0.2479, loss: 0.4196
2021-12-21 22:07:42,387 - mmdet - INFO - Epoch [4][100/930]     lr: 2.000e-02, eta: 0:33:54, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0207, loss_rpn_bbox: 0.0064, loss_cls: 0.1056, acc: 97.5508, loss_bbox: 0.0485, loss_mask: 0.2601, loss: 0.4413
2021-12-21 22:07:55,251 - mmdet - INFO - Epoch [4][150/930]     lr: 2.000e-02, eta: 0:33:43, time: 0.258, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0237, loss_rpn_bbox: 0.0062, loss_cls: 0.1223, acc: 97.2949, loss_bbox: 0.0547, loss_mask: 0.2307, loss: 0.4376
2021-12-21 22:08:07,899 - mmdet - INFO - Epoch [4][200/930]     lr: 2.000e-02, eta: 0:33:32, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0178, loss_rpn_bbox: 0.0056, loss_cls: 0.1065, acc: 97.6113, loss_bbox: 0.0489, loss_mask: 0.2554, loss: 0.4342
2021-12-21 22:08:20,486 - mmdet - INFO - Epoch [4][250/930]     lr: 2.000e-02, eta: 0:33:20, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0219, loss_rpn_bbox: 0.0059, loss_cls: 0.1035, acc: 97.5879, loss_bbox: 0.0446, loss_mask: 0.2369, loss: 0.4128
2021-12-21 22:08:32,865 - mmdet - INFO - Epoch [4][300/930]     lr: 2.000e-02, eta: 0:33:08, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0170, loss_rpn_bbox: 0.0064, loss_cls: 0.0853, acc: 98.1484, loss_bbox: 0.0361, loss_mask: 0.2394, loss: 0.3842
2021-12-21 22:08:45,194 - mmdet - INFO - Epoch [4][350/930]     lr: 2.000e-02, eta: 0:32:56, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0150, loss_rpn_bbox: 0.0055, loss_cls: 0.1072, acc: 97.4766, loss_bbox: 0.0466, loss_mask: 0.2238, loss: 0.3981
2021-12-21 22:08:57,715 - mmdet - INFO - Epoch [4][400/930]     lr: 2.000e-02, eta: 0:32:44, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0186, loss_rpn_bbox: 0.0059, loss_cls: 0.0992, acc: 97.7129, loss_bbox: 0.0450, loss_mask: 0.2251, loss: 0.3938
2021-12-21 22:09:10,325 - mmdet - INFO - Epoch [4][450/930]     lr: 2.000e-02, eta: 0:32:32, time: 0.253, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0191, loss_rpn_bbox: 0.0057, loss_cls: 0.0976, acc: 97.5684, loss_bbox: 0.0424, loss_mask: 0.2057, loss: 0.3705
2021-12-21 22:09:23,173 - mmdet - INFO - Epoch [4][500/930]     lr: 2.000e-02, eta: 0:32:21, time: 0.257, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0052, loss_cls: 0.1065, acc: 97.6270, loss_bbox: 0.0464, loss_mask: 0.2253, loss: 0.4005
2021-12-21 22:09:35,986 - mmdet - INFO - Epoch [4][550/930]     lr: 2.000e-02, eta: 0:32:10, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0180, loss_rpn_bbox: 0.0055, loss_cls: 0.0964, acc: 97.7637, loss_bbox: 0.0405, loss_mask: 0.2360, loss: 0.3964
2021-12-21 22:09:48,288 - mmdet - INFO - Epoch [4][600/930]     lr: 2.000e-02, eta: 0:31:58, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0169, loss_rpn_bbox: 0.0061, loss_cls: 0.0825, acc: 97.9883, loss_bbox: 0.0356, loss_mask: 0.2219, loss: 0.3630
2021-12-21 22:10:00,893 - mmdet - INFO - Epoch [4][650/930]     lr: 2.000e-02, eta: 0:31:46, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0152, loss_rpn_bbox: 0.0055, loss_cls: 0.0949, acc: 97.8418, loss_bbox: 0.0399, loss_mask: 0.2404, loss: 0.3959
2021-12-21 22:10:13,555 - mmdet - INFO - Epoch [4][700/930]     lr: 2.000e-02, eta: 0:31:34, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0132, loss_rpn_bbox: 0.0054, loss_cls: 0.1011, acc: 97.6289, loss_bbox: 0.0451, loss_mask: 0.2521, loss: 0.4168
2021-12-21 22:10:25,970 - mmdet - INFO - Epoch [4][750/930]     lr: 2.000e-02, eta: 0:31:22, time: 0.249, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0166, loss_rpn_bbox: 0.0056, loss_cls: 0.1006, acc: 97.7383, loss_bbox: 0.0450, loss_mask: 0.2427, loss: 0.4105
2021-12-21 22:10:38,441 - mmdet - INFO - Epoch [4][800/930]     lr: 2.000e-02, eta: 0:31:10, time: 0.249, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0166, loss_rpn_bbox: 0.0064, loss_cls: 0.0940, acc: 97.9082, loss_bbox: 0.0398, loss_mask: 0.2325, loss: 0.3893
2021-12-21 22:10:50,866 - mmdet - INFO - Epoch [4][850/930]     lr: 2.000e-02, eta: 0:30:58, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0187, loss_rpn_bbox: 0.0051, loss_cls: 0.0936, acc: 97.8340, loss_bbox: 0.0418, loss_mask: 0.2227, loss: 0.3818
2021-12-21 22:11:03,546 - mmdet - INFO - Epoch [4][900/930]     lr: 2.000e-02, eta: 0:30:46, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0182, loss_rpn_bbox: 0.0060, loss_cls: 0.0890, acc: 97.9160, loss_bbox: 0.0399, loss_mask: 0.2405, loss: 0.3936
2021-12-21 22:11:11,150 - mmdet - INFO - Saving checkpoint at 4 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.6 task/s, elapsed: 89s, ETA:     0s2021-12-21 22:12:41,740 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.60s).
Accumulating evaluation results...
DONE (t=0.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.014
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.010
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.210
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.130
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.130
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.130
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.130
2021-12-21 22:12:42,887 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.10s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.76s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.014
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.010
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.132
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.132
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.132
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.163
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.130
2021-12-21 22:12:44,367 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:12:44,368 - mmdet - INFO - Epoch(val) [4][764]    bbox_mAP: 0.0090, bbox_mAP_50: 0.0140, bbox_mAP_75: 0.0100, bbox_mAP_s: 0.2100, bbox_mAP_m: 0.0110, bbox_mAP_l: 0.0100, bbox_mAP_copypaste: 0.009 0.014 0.010 0.210 0.011 0.010, segm_mAP: 0.0090, segm_mAP_50: 0.0140, segm_mAP_75: 0.0100, segm_mAP_s: 0.2350, segm_mAP_m: 0.0110, segm_mAP_l: 0.0100, segm_mAP_copypaste: 0.009 0.014 0.010 0.235 0.011 0.010
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:5**

```
2021-12-21 22:12:59,256 - mmdet - INFO - Epoch [5][50/930]      lr: 2.000e-02, eta: 0:30:16, time: 0.296, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0191, loss_rpn_bbox: 0.0051, loss_cls: 0.0997, acc: 97.7422, loss_bbox: 0.0452, loss_mask: 0.2399, loss: 0.4090
2021-12-21 22:13:11,650 - mmdet - INFO - Epoch [5][100/930]     lr: 2.000e-02, eta: 0:30:04, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0164, loss_rpn_bbox: 0.0057, loss_cls: 0.0967, acc: 97.7012, loss_bbox: 0.0424, loss_mask: 0.2309, loss: 0.3921
2021-12-21 22:13:23,842 - mmdet - INFO - Epoch [5][150/930]     lr: 2.000e-02, eta: 0:29:52, time: 0.244, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0144, loss_rpn_bbox: 0.0050, loss_cls: 0.0992, acc: 97.5312, loss_bbox: 0.0472, loss_mask: 0.2333, loss: 0.3991
2021-12-21 22:13:36,225 - mmdet - INFO - Epoch [5][200/930]     lr: 2.000e-02, eta: 0:29:40, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0183, loss_rpn_bbox: 0.0059, loss_cls: 0.0865, acc: 97.9922, loss_bbox: 0.0403, loss_mask: 0.2443, loss: 0.3953
2021-12-21 22:13:48,706 - mmdet - INFO - Epoch [5][250/930]     lr: 2.000e-02, eta: 0:29:28, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0185, loss_rpn_bbox: 0.0049, loss_cls: 0.1027, acc: 97.5312, loss_bbox: 0.0468, loss_mask: 0.2309, loss: 0.4038
2021-12-21 22:14:01,050 - mmdet - INFO - Epoch [5][300/930]     lr: 2.000e-02, eta: 0:29:15, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0196, loss_rpn_bbox: 0.0060, loss_cls: 0.1024, acc: 97.5898, loss_bbox: 0.0477, loss_mask: 0.2306, loss: 0.4063
2021-12-21 22:14:13,427 - mmdet - INFO - Epoch [5][350/930]     lr: 2.000e-02, eta: 0:29:03, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0252, loss_rpn_bbox: 0.0060, loss_cls: 0.0962, acc: 97.8164, loss_bbox: 0.0426, loss_mask: 0.2425, loss: 0.4127
2021-12-21 22:14:25,975 - mmdet - INFO - Epoch [5][400/930]     lr: 2.000e-02, eta: 0:28:51, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0195, loss_rpn_bbox: 0.0056, loss_cls: 0.0958, acc: 97.7402, loss_bbox: 0.0417, loss_mask: 0.2165, loss: 0.3791
2021-12-21 22:14:38,588 - mmdet - INFO - Epoch [5][450/930]     lr: 2.000e-02, eta: 0:28:40, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0205, loss_rpn_bbox: 0.0056, loss_cls: 0.0856, acc: 98.0000, loss_bbox: 0.0375, loss_mask: 0.2246, loss: 0.3738
2021-12-21 22:14:51,315 - mmdet - INFO - Epoch [5][500/930]     lr: 2.000e-02, eta: 0:28:28, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0055, loss_cls: 0.1020, acc: 97.4434, loss_bbox: 0.0452, loss_mask: 0.2212, loss: 0.3910
2021-12-21 22:15:04,086 - mmdet - INFO - Epoch [5][550/930]     lr: 2.000e-02, eta: 0:28:16, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0048, loss_cls: 0.0919, acc: 97.7285, loss_bbox: 0.0394, loss_mask: 0.2166, loss: 0.3674
2021-12-21 22:15:16,677 - mmdet - INFO - Epoch [5][600/930]     lr: 2.000e-02, eta: 0:28:05, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0147, loss_rpn_bbox: 0.0055, loss_cls: 0.1034, acc: 97.4531, loss_bbox: 0.0476, loss_mask: 0.2292, loss: 0.4004
2021-12-21 22:15:29,072 - mmdet - INFO - Epoch [5][650/930]     lr: 2.000e-02, eta: 0:27:52, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0323, loss_rpn_bbox: 0.0074, loss_cls: 0.0825, acc: 98.0957, loss_bbox: 0.0370, loss_mask: 0.2301, loss: 0.3893
2021-12-21 22:15:41,725 - mmdet - INFO - Epoch [5][700/930]     lr: 2.000e-02, eta: 0:27:41, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0182, loss_rpn_bbox: 0.0059, loss_cls: 0.0942, acc: 97.8145, loss_bbox: 0.0419, loss_mask: 0.2305, loss: 0.3906
2021-12-21 22:15:54,314 - mmdet - INFO - Epoch [5][750/930]     lr: 2.000e-02, eta: 0:27:29, time: 0.252, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0152, loss_rpn_bbox: 0.0053, loss_cls: 0.1061, acc: 97.5020, loss_bbox: 0.0461, loss_mask: 0.2203, loss: 0.3930
2021-12-21 22:16:07,084 - mmdet - INFO - Epoch [5][800/930]     lr: 2.000e-02, eta: 0:27:17, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0046, loss_cls: 0.1110, acc: 97.2949, loss_bbox: 0.0472, loss_mask: 0.2044, loss: 0.3820
2021-12-21 22:16:19,554 - mmdet - INFO - Epoch [5][850/930]     lr: 2.000e-02, eta: 0:27:05, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0184, loss_rpn_bbox: 0.0056, loss_cls: 0.0883, acc: 97.8887, loss_bbox: 0.0375, loss_mask: 0.2192, loss: 0.3690
2021-12-21 22:16:32,420 - mmdet - INFO - Epoch [5][900/930]     lr: 2.000e-02, eta: 0:26:53, time: 0.257, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0165, loss_rpn_bbox: 0.0050, loss_cls: 0.0910, acc: 97.7402, loss_bbox: 0.0391, loss_mask: 0.2234, loss: 0.3751
2021-12-21 22:16:40,002 - mmdet - INFO - Saving checkpoint at 5 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 7.0 task/s, elapsed: 109s, ETA:     0s2021-12-21 22:18:31,847 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.16s).
Accumulating evaluation results...
DONE (t=0.54s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.024
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.038
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.021
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.248
2021-12-21 22:18:33,623 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.19s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.40s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.57s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.025
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.038
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.024
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.247
2021-12-21 22:18:35,971 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:18:35,972 - mmdet - INFO - Epoch(val) [5][764]    bbox_mAP: 0.0240, bbox_mAP_50: 0.0380, bbox_mAP_75: 0.0300, bbox_mAP_s: 0.3000, bbox_mAP_m: 0.0290, bbox_mAP_l: 0.0210, bbox_mAP_copypaste: 0.024 0.038 0.030 0.300 0.029 0.021, segm_mAP: 0.0250, segm_mAP_50: 0.0380, segm_mAP_75: 0.0300, segm_mAP_s: 0.1500, segm_mAP_m: 0.0300, segm_mAP_l: 0.0240, segm_mAP_copypaste: 0.025 0.038 0.030 0.150 0.030 0.024
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:6**

```
2021-12-21 22:18:50,938 - mmdet - INFO - Epoch [6][50/930]      lr: 2.000e-02, eta: 0:26:27, time: 0.297, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0178, loss_rpn_bbox: 0.0054, loss_cls: 0.1046, acc: 97.5000, loss_bbox: 0.0465, loss_mask: 0.2345, loss: 0.4088
2021-12-21 22:19:03,455 - mmdet - INFO - Epoch [6][100/930]     lr: 2.000e-02, eta: 0:26:15, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0056, loss_cls: 0.1022, acc: 97.3965, loss_bbox: 0.0459, loss_mask: 0.2005, loss: 0.3690
2021-12-21 22:19:15,698 - mmdet - INFO - Epoch [6][150/930]     lr: 2.000e-02, eta: 0:26:03, time: 0.245, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0158, loss_rpn_bbox: 0.0057, loss_cls: 0.1035, acc: 97.5293, loss_bbox: 0.0461, loss_mask: 0.2172, loss: 0.3883
2021-12-21 22:19:28,317 - mmdet - INFO - Epoch [6][200/930]     lr: 2.000e-02, eta: 0:25:51, time: 0.252, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0150, loss_rpn_bbox: 0.0048, loss_cls: 0.1070, acc: 97.3379, loss_bbox: 0.0469, loss_mask: 0.2191, loss: 0.3927
2021-12-21 22:19:40,845 - mmdet - INFO - Epoch [6][250/930]     lr: 2.000e-02, eta: 0:25:39, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0060, loss_cls: 0.1022, acc: 97.5742, loss_bbox: 0.0461, loss_mask: 0.2306, loss: 0.4021
2021-12-21 22:19:53,236 - mmdet - INFO - Epoch [6][300/930]     lr: 2.000e-02, eta: 0:25:27, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0138, loss_rpn_bbox: 0.0058, loss_cls: 0.1038, acc: 97.3652, loss_bbox: 0.0478, loss_mask: 0.2204, loss: 0.3916
2021-12-21 22:20:05,691 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:20:05,691 - mmdet - INFO - Epoch [6][350/930]     lr: 2.000e-02, eta: 0:25:14, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0131, loss_rpn_bbox: 0.0051, loss_cls: 0.1020, acc: 97.5039, loss_bbox: 0.0435, loss_mask: 0.2195, loss: 0.3832
2021-12-21 22:20:18,199 - mmdet - INFO - Epoch [6][400/930]     lr: 2.000e-02, eta: 0:25:02, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0154, loss_rpn_bbox: 0.0054, loss_cls: 0.1036, acc: 97.5762, loss_bbox: 0.0446, loss_mask: 0.2225, loss: 0.3914
2021-12-21 22:20:30,960 - mmdet - INFO - Epoch [6][450/930]     lr: 2.000e-02, eta: 0:24:51, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0144, loss_rpn_bbox: 0.0055, loss_cls: 0.1124, acc: 97.2754, loss_bbox: 0.0479, loss_mask: 0.2055, loss: 0.3857
2021-12-21 22:20:43,766 - mmdet - INFO - Epoch [6][500/930]     lr: 2.000e-02, eta: 0:24:39, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0054, loss_cls: 0.1029, acc: 97.3730, loss_bbox: 0.0492, loss_mask: 0.2205, loss: 0.3928
2021-12-21 22:20:56,749 - mmdet - INFO - Epoch [6][550/930]     lr: 2.000e-02, eta: 0:24:27, time: 0.260, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0124, loss_rpn_bbox: 0.0046, loss_cls: 0.0929, acc: 97.6836, loss_bbox: 0.0393, loss_mask: 0.1980, loss: 0.3472
2021-12-21 22:21:09,394 - mmdet - INFO - Epoch [6][600/930]     lr: 2.000e-02, eta: 0:24:15, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0213, loss_rpn_bbox: 0.0058, loss_cls: 0.0870, acc: 97.9023, loss_bbox: 0.0394, loss_mask: 0.2481, loss: 0.4016
2021-12-21 22:21:22,201 - mmdet - INFO - Epoch [6][650/930]     lr: 2.000e-02, eta: 0:24:04, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0195, loss_rpn_bbox: 0.0059, loss_cls: 0.0910, acc: 97.8125, loss_bbox: 0.0392, loss_mask: 0.2169, loss: 0.3724
2021-12-21 22:21:35,154 - mmdet - INFO - Epoch [6][700/930]     lr: 2.000e-02, eta: 0:23:52, time: 0.259, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0177, loss_rpn_bbox: 0.0054, loss_cls: 0.0969, acc: 97.6523, loss_bbox: 0.0422, loss_mask: 0.2331, loss: 0.3953

2021-12-21 22:21:47,639 - mmdet - INFO - Epoch [6][750/930]     lr: 2.000e-02, eta: 0:23:40, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0160, loss_rpn_bbox: 0.0051, loss_cls: 0.0972, acc: 97.6855, loss_bbox: 0.0432, loss_mask: 0.2231, loss: 0.3845
2021-12-21 22:22:00,345 - mmdet - INFO - Epoch [6][800/930]     lr: 2.000e-02, eta: 0:23:28, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0175, loss_rpn_bbox: 0.0054, loss_cls: 0.0903, acc: 97.8496, loss_bbox: 0.0404, loss_mask: 0.2247, loss: 0.3784
2021-12-21 22:22:12,650 - mmdet - INFO - Epoch [6][850/930]     lr: 2.000e-02, eta: 0:23:16, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0190, loss_rpn_bbox: 0.0052, loss_cls: 0.1004, acc: 97.7363, loss_bbox: 0.0446, loss_mask: 0.2377, loss: 0.4069
2021-12-21 22:22:25,443 - mmdet - INFO - Epoch [6][900/930]     lr: 2.000e-02, eta: 0:23:04, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0066, loss_cls: 0.1083, acc: 97.2910, loss_bbox: 0.0480, loss_mask: 0.2230, loss: 0.4031
2021-12-21 22:22:33,073 - mmdet - INFO - Saving checkpoint at 6 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 6.7 task/s, elapsed: 114s, ETA:     0s2021-12-21 22:24:29,284 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.92s).
Accumulating evaluation results...
DONE (t=0.50s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.029
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.022
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.024
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.017
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.273
2021-12-21 22:24:30,872 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.17s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.11s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.52s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.029
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.023
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.026
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.268
2021-12-21 22:24:32,890 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:24:32,890 - mmdet - INFO - Epoch(val) [6][764]    bbox_mAP: 0.0190, bbox_mAP_50: 0.0290, bbox_mAP_75: 0.0220, bbox_mAP_s: 0.3040, bbox_mAP_m: 0.0240, bbox_mAP_l: 0.0170, bbox_mAP_copypaste: 0.019 0.029 0.022 0.304 0.024 0.017, segm_mAP: 0.0180, segm_mAP_50: 0.0290, segm_mAP_75: 0.0210, segm_mAP_s: 0.2370, segm_mAP_m: 0.0230, segm_mAP_l: 0.0260, segm_mAP_copypaste: 0.018 0.029 0.021 0.237 0.023 0.026
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:7**

```
2021-12-21 22:24:47,860 - mmdet - INFO - Epoch [7][50/930]      lr: 2.000e-02, eta: 0:22:39, time: 0.298, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0132, loss_rpn_bbox: 0.0048, loss_cls: 0.0987, acc: 97.4902, loss_bbox: 0.0436, loss_mask: 0.1922, loss: 0.3525
2021-12-21 22:25:00,237 - mmdet - INFO - Epoch [7][100/930]     lr: 2.000e-02, eta: 0:22:27, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0168, loss_rpn_bbox: 0.0047, loss_cls: 0.1023, acc: 97.5000, loss_bbox: 0.0438, loss_mask: 0.1922, loss: 0.3597
2021-12-21 22:25:12,577 - mmdet - INFO - Epoch [7][150/930]     lr: 2.000e-02, eta: 0:22:15, time: 0.247, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0161, loss_rpn_bbox: 0.0059, loss_cls: 0.1027, acc: 97.3711, loss_bbox: 0.0461, loss_mask: 0.2093, loss: 0.3802
2021-12-21 22:25:25,143 - mmdet - INFO - Epoch [7][200/930]     lr: 2.000e-02, eta: 0:22:03, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0154, loss_rpn_bbox: 0.0055, loss_cls: 0.0967, acc: 97.5684, loss_bbox: 0.0439, loss_mask: 0.2139, loss: 0.3755
2021-12-21 22:25:37,946 - mmdet - INFO - Epoch [7][250/930]     lr: 2.000e-02, eta: 0:21:51, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0185, loss_rpn_bbox: 0.0056, loss_cls: 0.0965, acc: 97.6582, loss_bbox: 0.0438, loss_mask: 0.2308, loss: 0.3952
2021-12-21 22:25:50,514 - mmdet - INFO - Epoch [7][300/930]     lr: 2.000e-02, eta: 0:21:39, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0155, loss_rpn_bbox: 0.0049, loss_cls: 0.0956, acc: 97.6973, loss_bbox: 0.0410, loss_mask: 0.2051, loss: 0.3620
2021-12-21 22:26:02,964 - mmdet - INFO - Epoch [7][350/930]     lr: 2.000e-02, eta: 0:21:27, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0135, loss_rpn_bbox: 0.0045, loss_cls: 0.1103, acc: 97.3008, loss_bbox: 0.0486, loss_mask: 0.2142, loss: 0.3911
2021-12-21 22:26:15,469 - mmdet - INFO - Epoch [7][400/930]     lr: 2.000e-02, eta: 0:21:14, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0161, loss_rpn_bbox: 0.0047, loss_cls: 0.0951, acc: 97.5859, loss_bbox: 0.0372, loss_mask: 0.1893, loss: 0.3424
2021-12-21 22:26:27,994 - mmdet - INFO - Epoch [7][450/930]     lr: 2.000e-02, eta: 0:21:02, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0165, loss_rpn_bbox: 0.0055, loss_cls: 0.0944, acc: 97.6973, loss_bbox: 0.0428, loss_mask: 0.2330, loss: 0.3922
2021-12-21 22:26:40,694 - mmdet - INFO - Epoch [7][500/930]     lr: 2.000e-02, eta: 0:20:50, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0161, loss_rpn_bbox: 0.0053, loss_cls: 0.1006, acc: 97.6211, loss_bbox: 0.0455, loss_mask: 0.2200, loss: 0.3874
2021-12-21 22:26:53,652 - mmdet - INFO - Epoch [7][550/930]     lr: 2.000e-02, eta: 0:20:39, time: 0.259, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0139, loss_rpn_bbox: 0.0049, loss_cls: 0.1103, acc: 97.2812, loss_bbox: 0.0500, loss_mask: 0.2271, loss: 0.4062
2021-12-21 22:27:06,094 - mmdet - INFO - Epoch [7][600/930]     lr: 2.000e-02, eta: 0:20:26, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0208, loss_rpn_bbox: 0.0063, loss_cls: 0.0957, acc: 97.6699, loss_bbox: 0.0418, loss_mask: 0.2276, loss: 0.3922
2021-12-21 22:27:18,668 - mmdet - INFO - Epoch [7][650/930]     lr: 2.000e-02, eta: 0:20:14, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0135, loss_rpn_bbox: 0.0044, loss_cls: 0.1073, acc: 97.1836, loss_bbox: 0.0506, loss_mask: 0.2022, loss: 0.3779
2021-12-21 22:27:31,451 - mmdet - INFO - Epoch [7][700/930]     lr: 2.000e-02, eta: 0:20:02, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0051, loss_cls: 0.1092, acc: 97.3652, loss_bbox: 0.0532, loss_mask: 0.2371, loss: 0.4165
2021-12-21 22:27:44,188 - mmdet - INFO - Epoch [7][750/930]     lr: 2.000e-02, eta: 0:19:50, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0054, loss_cls: 0.1016, acc: 97.6230, loss_bbox: 0.0465, loss_mask: 0.2316, loss: 0.3998
2021-12-21 22:27:57,153 - mmdet - INFO - Epoch [7][800/930]     lr: 2.000e-02, eta: 0:19:38, time: 0.259, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0135, loss_rpn_bbox: 0.0053, loss_cls: 0.1181, acc: 97.0820, loss_bbox: 0.0499, loss_mask: 0.2011, loss: 0.3879
2021-12-21 22:28:09,665 - mmdet - INFO - Epoch [7][850/930]     lr: 2.000e-02, eta: 0:19:26, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0173, loss_rpn_bbox: 0.0054, loss_cls: 0.1087, acc: 97.3691, loss_bbox: 0.0463, loss_mask: 0.2208, loss: 0.3985
2021-12-21 22:28:22,625 - mmdet - INFO - Epoch [7][900/930]     lr: 2.000e-02, eta: 0:19:14, time: 0.259, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0161, loss_rpn_bbox: 0.0058, loss_cls: 0.0965, acc: 97.6387, loss_bbox: 0.0439, loss_mask: 0.2280, loss: 0.3903
2021-12-21 22:28:30,260 - mmdet - INFO - Saving checkpoint at 7 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.5 task/s, elapsed: 90s, ETA:     0s2021-12-21 22:30:02,842 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.89s).
Accumulating evaluation results...
DONE (t=0.39s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.039
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.023
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.270
2021-12-21 22:30:04,167 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.97s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.038
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.087
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.023
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.269
2021-12-21 22:30:05,919 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:30:05,919 - mmdet - INFO - Epoch(val) [7][764]    bbox_mAP: 0.0270, bbox_mAP_50: 0.0390, bbox_mAP_75: 0.0310, bbox_mAP_s: 0.2250, bbox_mAP_m: 0.0320, bbox_mAP_l: 0.0230, bbox_mAP_copypaste: 0.027 0.039 0.031 0.225 0.032 0.023, segm_mAP: 0.0270, segm_mAP_50: 0.0380, segm_mAP_75: 0.0300, segm_mAP_s: 0.0870, segm_mAP_m: 0.0320, segm_mAP_l: 0.0230, segm_mAP_copypaste: 0.027 0.038 0.030 0.087 0.032 0.023
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:8**

```
2021-12-21 22:30:20,980 - mmdet - INFO - Epoch [8][50/930]      lr: 2.000e-02, eta: 0:18:51, time: 0.299, data_time: 0.053, memory: 3050, loss_rpn_cls: 0.0149, loss_rpn_bbox: 0.0053, loss_cls: 0.1021, acc: 97.3457, loss_bbox: 0.0456, loss_mask: 0.2128, loss: 0.3807
2021-12-21 22:30:33,420 - mmdet - INFO - Epoch [8][100/930]     lr: 2.000e-02, eta: 0:18:39, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0140, loss_rpn_bbox: 0.0049, loss_cls: 0.1059, acc: 97.4062, loss_bbox: 0.0445, loss_mask: 0.2078, loss: 0.3771
2021-12-21 22:30:45,825 - mmdet - INFO - Epoch [8][150/930]     lr: 2.000e-02, eta: 0:18:27, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0140, loss_rpn_bbox: 0.0050, loss_cls: 0.1086, acc: 97.1602, loss_bbox: 0.0465, loss_mask: 0.1993, loss: 0.3734
2021-12-21 22:30:58,505 - mmdet - INFO - Epoch [8][200/930]     lr: 2.000e-02, eta: 0:18:15, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0141, loss_rpn_bbox: 0.0044, loss_cls: 0.1020, acc: 97.5938, loss_bbox: 0.0430, loss_mask: 0.2114, loss: 0.3750
2021-12-21 22:31:11,166 - mmdet - INFO - Epoch [8][250/930]     lr: 2.000e-02, eta: 0:18:03, time: 0.254, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0137, loss_rpn_bbox: 0.0051, loss_cls: 0.0938, acc: 97.5723, loss_bbox: 0.0396, loss_mask: 0.2014, loss: 0.3535
2021-12-21 22:31:23,590 - mmdet - INFO - Epoch [8][300/930]     lr: 2.000e-02, eta: 0:17:50, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0191, loss_rpn_bbox: 0.0052, loss_cls: 0.0983, acc: 97.6641, loss_bbox: 0.0435, loss_mask: 0.2367, loss: 0.4028
2021-12-21 22:31:35,945 - mmdet - INFO - Epoch [8][350/930]     lr: 2.000e-02, eta: 0:17:38, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0154, loss_rpn_bbox: 0.0056, loss_cls: 0.0983, acc: 97.5098, loss_bbox: 0.0455, loss_mask: 0.2168, loss: 0.3817
2021-12-21 22:31:48,742 - mmdet - INFO - Epoch [8][400/930]     lr: 2.000e-02, eta: 0:17:26, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0136, loss_rpn_bbox: 0.0049, loss_cls: 0.0988, acc: 97.6621, loss_bbox: 0.0427, loss_mask: 0.2272, loss: 0.3872
2021-12-21 22:32:01,347 - mmdet - INFO - Epoch [8][450/930]     lr: 2.000e-02, eta: 0:17:14, time: 0.252, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0167, loss_rpn_bbox: 0.0052, loss_cls: 0.1032, acc: 97.4922, loss_bbox: 0.0453, loss_mask: 0.2312, loss: 0.4017
2021-12-21 22:32:14,070 - mmdet - INFO - Epoch [8][500/930]     lr: 2.000e-02, eta: 0:17:02, time: 0.254, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0184, loss_rpn_bbox: 0.0049, loss_cls: 0.1044, acc: 97.5000, loss_bbox: 0.0508, loss_mask: 0.2461, loss: 0.4246
2021-12-21 22:32:27,022 - mmdet - INFO - Epoch [8][550/930]     lr: 2.000e-02, eta: 0:16:50, time: 0.259, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0159, loss_rpn_bbox: 0.0043, loss_cls: 0.1045, acc: 97.3262, loss_bbox: 0.0462, loss_mask: 0.1941, loss: 0.3650
2021-12-21 22:32:39,525 - mmdet - INFO - Epoch [8][600/930]     lr: 2.000e-02, eta: 0:16:38, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0160, loss_rpn_bbox: 0.0050, loss_cls: 0.1025, acc: 97.5410, loss_bbox: 0.0443, loss_mask: 0.2055, loss: 0.3732
2021-12-21 22:32:52,152 - mmdet - INFO - Epoch [8][650/930]     lr: 2.000e-02, eta: 0:16:26, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0211, loss_rpn_bbox: 0.0059, loss_cls: 0.0970, acc: 97.6543, loss_bbox: 0.0429, loss_mask: 0.2143, loss: 0.3813
2021-12-21 22:33:05,200 - mmdet - INFO - Epoch [8][700/930]     lr: 2.000e-02, eta: 0:16:14, time: 0.261, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0137, loss_rpn_bbox: 0.0056, loss_cls: 0.0976, acc: 97.5801, loss_bbox: 0.0459, loss_mask: 0.2218, loss: 0.3847
2021-12-21 22:33:17,732 - mmdet - INFO - Epoch [8][750/930]     lr: 2.000e-02, eta: 0:16:01, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0142, loss_rpn_bbox: 0.0052, loss_cls: 0.0913, acc: 97.7148, loss_bbox: 0.0369, loss_mask: 0.1966, loss: 0.3442
2021-12-21 22:33:30,221 - mmdet - INFO - Epoch [8][800/930]     lr: 2.000e-02, eta: 0:15:49, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0129, loss_rpn_bbox: 0.0051, loss_cls: 0.0923, acc: 97.6328, loss_bbox: 0.0396, loss_mask: 0.2022, loss: 0.3521
2021-12-21 22:33:42,702 - mmdet - INFO - Epoch [8][850/930]     lr: 2.000e-02, eta: 0:15:37, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0190, loss_rpn_bbox: 0.0061, loss_cls: 0.0884, acc: 97.8340, loss_bbox: 0.0392, loss_mask: 0.2294, loss: 0.3821
2021-12-21 22:33:55,612 - mmdet - INFO - Epoch [8][900/930]     lr: 2.000e-02, eta: 0:15:25, time: 0.258, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0150, loss_rpn_bbox: 0.0057, loss_cls: 0.0916, acc: 97.5996, loss_bbox: 0.0394, loss_mask: 0.2000, loss: 0.3517
2021-12-21 22:34:03,423 - mmdet - INFO - Saving checkpoint at 8 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.1 task/s, elapsed: 94s, ETA:     0s2021-12-21 22:35:39,499 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.11s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.80s).
Accumulating evaluation results...
DONE (t=0.42s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.025
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.017
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.407
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.268
2021-12-21 22:35:40,870 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.98s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.42s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.366
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.274
2021-12-21 22:35:42,635 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:35:42,635 - mmdet - INFO - Epoch(val) [8][764]    bbox_mAP: 0.0250, bbox_mAP_50: 0.0400, bbox_mAP_75: 0.0300, bbox_mAP_s: 0.2370, bbox_mAP_m: 0.0310, bbox_mAP_l: 0.0170, bbox_mAP_copypaste: 0.025 0.040 0.030 0.237 0.031 0.017, segm_mAP: 0.0280, segm_mAP_50: 0.0400, segm_mAP_75: 0.0320, segm_mAP_s: 0.2370, segm_mAP_m: 0.0330, segm_mAP_l: 0.0200, segm_mAP_copypaste: 0.028 0.040 0.032 0.237 0.033 0.020
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:9**

```
2021-12-21 22:35:57,522 - mmdet - INFO - Epoch [9][50/930]      lr: 2.000e-03, eta: 0:15:03, time: 0.296, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0110, loss_rpn_bbox: 0.0041, loss_cls: 0.0940, acc: 97.5781, loss_bbox: 0.0395, loss_mask: 0.2024, loss: 0.3509
2021-12-21 22:36:10,052 - mmdet - INFO - Epoch [9][100/930]     lr: 2.000e-03, eta: 0:14:51, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0112, loss_rpn_bbox: 0.0045, loss_cls: 0.0944, acc: 97.4043, loss_bbox: 0.0372, loss_mask: 0.1804, loss: 0.3276
2021-12-21 22:36:22,360 - mmdet - INFO - Epoch [9][150/930]     lr: 2.000e-03, eta: 0:14:38, time: 0.246, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0049, loss_cls: 0.1019, acc: 97.2656, loss_bbox: 0.0443, loss_mask: 0.1931, loss: 0.3565
2021-12-21 22:36:35,031 - mmdet - INFO - Epoch [9][200/930]     lr: 2.000e-03, eta: 0:14:26, time: 0.253, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0094, loss_rpn_bbox: 0.0036, loss_cls: 0.1023, acc: 97.1973, loss_bbox: 0.0423, loss_mask: 0.1790, loss: 0.3365
2021-12-21 22:36:47,660 - mmdet - INFO - Epoch [9][250/930]     lr: 2.000e-03, eta: 0:14:14, time: 0.253, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0034, loss_cls: 0.1083, acc: 97.0371, loss_bbox: 0.0428, loss_mask: 0.1858, loss: 0.3501
2021-12-21 22:37:00,234 - mmdet - INFO - Epoch [9][300/930]     lr: 2.000e-03, eta: 0:14:02, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0104, loss_rpn_bbox: 0.0041, loss_cls: 0.1063, acc: 97.1562, loss_bbox: 0.0428, loss_mask: 0.1814, loss: 0.3449
2021-12-21 22:37:12,716 - mmdet - INFO - Epoch [9][350/930]     lr: 2.000e-03, eta: 0:13:50, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0100, loss_rpn_bbox: 0.0038, loss_cls: 0.1042, acc: 97.1699, loss_bbox: 0.0424, loss_mask: 0.1854, loss: 0.3458
2021-12-21 22:37:25,292 - mmdet - INFO - Epoch [9][400/930]     lr: 2.000e-03, eta: 0:13:37, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0124, loss_rpn_bbox: 0.0038, loss_cls: 0.1036, acc: 97.1895, loss_bbox: 0.0423, loss_mask: 0.1971, loss: 0.3592
2021-12-21 22:37:37,947 - mmdet - INFO - Epoch [9][450/930]     lr: 2.000e-03, eta: 0:13:25, time: 0.254, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0123, loss_rpn_bbox: 0.0039, loss_cls: 0.0996, acc: 97.1641, loss_bbox: 0.0381, loss_mask: 0.1716, loss: 0.3257
2021-12-21 22:37:50,760 - mmdet - INFO - Epoch [9][500/930]     lr: 2.000e-03, eta: 0:13:13, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0097, loss_rpn_bbox: 0.0036, loss_cls: 0.1028, acc: 97.1270, loss_bbox: 0.0434, loss_mask: 0.1839, loss: 0.3434
2021-12-21 22:38:03,627 - mmdet - INFO - Epoch [9][550/930]     lr: 2.000e-03, eta: 0:13:01, time: 0.257, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0114, loss_rpn_bbox: 0.0043, loss_cls: 0.1081, acc: 97.1582, loss_bbox: 0.0454, loss_mask: 0.2013, loss: 0.3704
2021-12-21 22:38:16,229 - mmdet - INFO - Epoch [9][600/930]     lr: 2.000e-03, eta: 0:12:49, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0042, loss_cls: 0.1063, acc: 97.0156, loss_bbox: 0.0430, loss_mask: 0.1809, loss: 0.3438
2021-12-21 22:38:29,096 - mmdet - INFO - Epoch [9][650/930]     lr: 2.000e-03, eta: 0:12:37, time: 0.257, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0041, loss_cls: 0.1046, acc: 97.1211, loss_bbox: 0.0427, loss_mask: 0.1833, loss: 0.3442
2021-12-21 22:38:42,033 - mmdet - INFO - Epoch [9][700/930]     lr: 2.000e-03, eta: 0:12:24, time: 0.259, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0108, loss_rpn_bbox: 0.0038, loss_cls: 0.0982, acc: 97.1562, loss_bbox: 0.0386, loss_mask: 0.1743, loss: 0.3257
2021-12-21 22:38:54,590 - mmdet - INFO - Epoch [9][750/930]     lr: 2.000e-03, eta: 0:12:12, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0109, loss_rpn_bbox: 0.0037, loss_cls: 0.0991, acc: 97.3496, loss_bbox: 0.0420, loss_mask: 0.2050, loss: 0.3607
2021-12-21 22:39:07,131 - mmdet - INFO - Epoch [9][800/930]     lr: 2.000e-03, eta: 0:12:00, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0106, loss_rpn_bbox: 0.0037, loss_cls: 0.0944, acc: 97.4316, loss_bbox: 0.0374, loss_mask: 0.1850, loss: 0.3311
2021-12-21 22:39:19,558 - mmdet - INFO - Epoch [9][850/930]     lr: 2.000e-03, eta: 0:11:48, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0038, loss_cls: 0.1005, acc: 97.2676, loss_bbox: 0.0445, loss_mask: 0.1911, loss: 0.3501
2021-12-21 22:39:32,365 - mmdet - INFO - Epoch [9][900/930]     lr: 2.000e-03, eta: 0:11:35, time: 0.256, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0133, loss_rpn_bbox: 0.0044, loss_cls: 0.0947, acc: 97.3730, loss_bbox: 0.0408, loss_mask: 0.2005, loss: 0.3536
2021-12-21 22:39:39,984 - mmdet - INFO - Saving checkpoint at 9 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.6 task/s, elapsed: 89s, ETA:     0s2021-12-21 22:41:10,973 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.13s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.81s).
Accumulating evaluation results...
DONE (t=0.41s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.291
2021-12-21 22:41:12,357 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.11s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.92s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.44s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.039
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.287
2021-12-21 22:41:14,077 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:41:14,077 - mmdet - INFO - Epoch(val) [9][764]    bbox_mAP: 0.0270, bbox_mAP_50: 0.0400, bbox_mAP_75: 0.0310, bbox_mAP_s: 0.2400, bbox_mAP_m: 0.0320, bbox_mAP_l: 0.0220, bbox_mAP_copypaste: 0.027 0.040 0.031 0.240 0.032 0.022, segm_mAP: 0.0270, segm_mAP_50: 0.0390, segm_mAP_75: 0.0320, segm_mAP_s: 0.0300, segm_mAP_m: 0.0310, segm_mAP_l: 0.0220, segm_mAP_copypaste: 0.027 0.039 0.032 0.030 0.031 0.022
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:10**

```
2021-12-21 22:41:29,152 - mmdet - INFO - Epoch [10][50/930]     lr: 2.000e-03, eta: 0:11:14, time: 0.300, data_time: 0.053, memory: 3050, loss_rpn_cls: 0.0075, loss_rpn_bbox: 0.0030, loss_cls: 0.0960, acc: 97.2598, loss_bbox: 0.0366, loss_mask: 0.1699, loss: 0.3130
2021-12-21 22:41:41,578 - mmdet - INFO - Epoch [10][100/930]    lr: 2.000e-03, eta: 0:11:02, time: 0.248, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0038, loss_cls: 0.1017, acc: 97.1523, loss_bbox: 0.0430, loss_mask: 0.1848, loss: 0.3438
2021-12-21 22:41:54,089 - mmdet - INFO - Epoch [10][150/930]    lr: 2.000e-03, eta: 0:10:50, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0078, loss_rpn_bbox: 0.0036, loss_cls: 0.1023, acc: 97.1797, loss_bbox: 0.0415, loss_mask: 0.1804, loss: 0.3356
2021-12-21 22:42:06,712 - mmdet - INFO - Epoch [10][200/930]    lr: 2.000e-03, eta: 0:10:37, time: 0.252, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0098, loss_rpn_bbox: 0.0033, loss_cls: 0.1024, acc: 97.0781, loss_bbox: 0.0405, loss_mask: 0.1844, loss: 0.3404
2021-12-21 22:42:19,179 - mmdet - INFO - Epoch [10][250/930]    lr: 2.000e-03, eta: 0:10:25, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0037, loss_cls: 0.0980, acc: 97.2031, loss_bbox: 0.0381, loss_mask: 0.1739, loss: 0.3223
2021-12-21 22:42:31,738 - mmdet - INFO - Epoch [10][300/930]    lr: 2.000e-03, eta: 0:10:13, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0092, loss_rpn_bbox: 0.0031, loss_cls: 0.0971, acc: 97.3457, loss_bbox: 0.0364, loss_mask: 0.1638, loss: 0.3096
2021-12-21 22:42:44,209 - mmdet - INFO - Epoch [10][350/930]    lr: 2.000e-03, eta: 0:10:01, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0083, loss_rpn_bbox: 0.0029, loss_cls: 0.0985, acc: 97.1777, loss_bbox: 0.0406, loss_mask: 0.1747, loss: 0.3249
2021-12-21 22:42:56,786 - mmdet - INFO - Epoch [10][400/930]    lr: 2.000e-03, eta: 0:09:48, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0098, loss_rpn_bbox: 0.0035, loss_cls: 0.0984, acc: 97.2188, loss_bbox: 0.0427, loss_mask: 0.1837, loss: 0.3381
2021-12-21 22:43:09,539 - mmdet - INFO - Epoch [10][450/930]    lr: 2.000e-03, eta: 0:09:36, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0080, loss_rpn_bbox: 0.0028, loss_cls: 0.1045, acc: 97.0723, loss_bbox: 0.0432, loss_mask: 0.1843, loss: 0.3428
2021-12-21 22:43:22,275 - mmdet - INFO - Epoch [10][500/930]    lr: 2.000e-03, eta: 0:09:24, time: 0.255, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0089, loss_rpn_bbox: 0.0035, loss_cls: 0.1032, acc: 97.1406, loss_bbox: 0.0420, loss_mask: 0.1768, loss: 0.3344
2021-12-21 22:43:35,294 - mmdet - INFO - Epoch [10][550/930]    lr: 2.000e-03, eta: 0:09:12, time: 0.261, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0033, loss_cls: 0.0964, acc: 97.3281, loss_bbox: 0.0342, loss_mask: 0.1658, loss: 0.3096
2021-12-21 22:43:47,783 - mmdet - INFO - Epoch [10][600/930]    lr: 2.000e-03, eta: 0:09:00, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0104, loss_rpn_bbox: 0.0031, loss_cls: 0.0975, acc: 97.2461, loss_bbox: 0.0424, loss_mask: 0.1929, loss: 0.3463
2021-12-21 22:44:00,380 - mmdet - INFO - Epoch [10][650/930]    lr: 2.000e-03, eta: 0:08:47, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0079, loss_rpn_bbox: 0.0038, loss_cls: 0.0927, acc: 97.3418, loss_bbox: 0.0327, loss_mask: 0.1578, loss: 0.2949
2021-12-21 22:44:13,144 - mmdet - INFO - Epoch [10][700/930]    lr: 2.000e-03, eta: 0:08:35, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0116, loss_rpn_bbox: 0.0036, loss_cls: 0.0974, acc: 97.3184, loss_bbox: 0.0369, loss_mask: 0.1717, loss: 0.3212
2021-12-21 22:44:25,743 - mmdet - INFO - Epoch [10][750/930]    lr: 2.000e-03, eta: 0:08:23, time: 0.252, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0117, loss_rpn_bbox: 0.0037, loss_cls: 0.0926, acc: 97.3027, loss_bbox: 0.0367, loss_mask: 0.1694, loss: 0.3142
2021-12-21 22:44:38,273 - mmdet - INFO - Epoch [10][800/930]    lr: 2.000e-03, eta: 0:08:10, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0098, loss_rpn_bbox: 0.0037, loss_cls: 0.1000, acc: 97.1914, loss_bbox: 0.0355, loss_mask: 0.1715, loss: 0.3205
2021-12-21 22:44:50,699 - mmdet - INFO - Epoch [10][850/930]    lr: 2.000e-03, eta: 0:07:58, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0038, loss_cls: 0.0987, acc: 97.2422, loss_bbox: 0.0386, loss_mask: 0.1738, loss: 0.3271
2021-12-21 22:45:03,582 - mmdet - INFO - Epoch [10][900/930]    lr: 2.000e-03, eta: 0:07:46, time: 0.258, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0108, loss_rpn_bbox: 0.0039, loss_cls: 0.0902, acc: 97.4922, loss_bbox: 0.0346, loss_mask: 0.1881, loss: 0.3277
2021-12-21 22:45:11,144 - mmdet - INFO - Saving checkpoint at 10 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.7 task/s, elapsed: 88s, ETA:     0s2021-12-21 22:46:41,036 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.03s).
Accumulating evaluation results...
DONE (t=0.41s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.026
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.292
2021-12-21 22:46:42,629 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.16s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.41s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.290
2021-12-21 22:46:44,436 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:46:44,437 - mmdet - INFO - Epoch(val) [10][764]   bbox_mAP: 0.0280, bbox_mAP_50: 0.0400, bbox_mAP_75: 0.0320, bbox_mAP_s: 0.3750, bbox_mAP_m: 0.0340, bbox_mAP_l: 0.0260, bbox_mAP_copypaste: 0.028 0.040 0.032 0.375 0.034 0.026, segm_mAP: 0.0290, segm_mAP_50: 0.0400, segm_mAP_75: 0.0320, segm_mAP_s: 0.2150, segm_mAP_m: 0.0340, segm_mAP_l: 0.0290, segm_mAP_copypaste: 0.029 0.040 0.032 0.215 0.034 0.029
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch:11**

```
2021-12-21 22:46:59,408 - mmdet - INFO - Epoch [11][50/930]     lr: 2.000e-03, eta: 0:07:25, time: 0.298, data_time: 0.052, memory: 3050, loss_rpn_cls: 0.0089, loss_rpn_bbox: 0.0034, loss_cls: 0.0933, acc: 97.3789, loss_bbox: 0.0343, loss_mask: 0.1670, loss: 0.3069
2021-12-21 22:47:11,794 - mmdet - INFO - Epoch [11][100/930]    lr: 2.000e-03, eta: 0:07:13, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0111, loss_rpn_bbox: 0.0031, loss_cls: 0.0959, acc: 97.3809, loss_bbox: 0.0396, loss_mask: 0.1914, loss: 0.3410
2021-12-21 22:47:24,298 - mmdet - INFO - Epoch [11][150/930]    lr: 2.000e-03, eta: 0:07:01, time: 0.250, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0093, loss_rpn_bbox: 0.0035, loss_cls: 0.0974, acc: 97.1621, loss_bbox: 0.0396, loss_mask: 0.1836, loss: 0.3335
2021-12-21 22:47:37,263 - mmdet - INFO - Epoch [11][200/930]    lr: 2.000e-03, eta: 0:06:48, time: 0.259, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0078, loss_rpn_bbox: 0.0036, loss_cls: 0.0989, acc: 97.1270, loss_bbox: 0.0365, loss_mask: 0.1588, loss: 0.3056
2021-12-21 22:47:49,826 - mmdet - INFO - Epoch [11][250/930]    lr: 2.000e-03, eta: 0:06:36, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0082, loss_rpn_bbox: 0.0032, loss_cls: 0.0950, acc: 97.2344, loss_bbox: 0.0390, loss_mask: 0.1812, loss: 0.3266
2021-12-21 22:48:02,369 - mmdet - INFO - Epoch [11][300/930]    lr: 2.000e-03, eta: 0:06:24, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0088, loss_rpn_bbox: 0.0035, loss_cls: 0.0988, acc: 97.1758, loss_bbox: 0.0386, loss_mask: 0.1627, loss: 0.3124
2021-12-21 22:48:14,901 - mmdet - INFO - Epoch [11][350/930]    lr: 2.000e-03, eta: 0:06:12, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0089, loss_rpn_bbox: 0.0034, loss_cls: 0.1009, acc: 97.1836, loss_bbox: 0.0401, loss_mask: 0.1852, loss: 0.3386
2021-12-21 22:48:27,535 - mmdet - INFO - Epoch [11][400/930]    lr: 2.000e-03, eta: 0:05:59, time: 0.253, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0107, loss_rpn_bbox: 0.0038, loss_cls: 0.0946, acc: 97.3086, loss_bbox: 0.0403, loss_mask: 0.1866, loss: 0.3359
2021-12-21 22:48:40,439 - mmdet - INFO - Epoch [11][450/930]    lr: 2.000e-03, eta: 0:05:47, time: 0.258, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0113, loss_rpn_bbox: 0.0035, loss_cls: 0.0928, acc: 97.2500, loss_bbox: 0.0379, loss_mask: 0.1665, loss: 0.3121
2021-12-21 22:48:53,302 - mmdet - INFO - Epoch [11][500/930]    lr: 2.000e-03, eta: 0:05:35, time: 0.257, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0081, loss_rpn_bbox: 0.0029, loss_cls: 0.0917, acc: 97.2832, loss_bbox: 0.0348, loss_mask: 0.1564, loss: 0.2939
2021-12-21 22:49:06,198 - mmdet - INFO - Epoch [11][550/930]    lr: 2.000e-03, eta: 0:05:23, time: 0.258, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0108, loss_rpn_bbox: 0.0032, loss_cls: 0.0909, acc: 97.4531, loss_bbox: 0.0354, loss_mask: 0.1742, loss: 0.3145
2021-12-21 22:49:18,938 - mmdet - INFO - Epoch [11][600/930]    lr: 2.000e-03, eta: 0:05:10, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0033, loss_cls: 0.0920, acc: 97.3613, loss_bbox: 0.0343, loss_mask: 0.1648, loss: 0.3047
2021-12-21 22:49:31,515 - mmdet - INFO - Epoch [11][650/930]    lr: 2.000e-03, eta: 0:04:58, time: 0.251, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0034, loss_cls: 0.0949, acc: 97.3184, loss_bbox: 0.0361, loss_mask: 0.1822, loss: 0.3264
2021-12-21 22:49:44,211 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:49:44,211 - mmdet - INFO - Epoch [11][700/930]    lr: 2.000e-03, eta: 0:04:46, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0080, loss_rpn_bbox: 0.0032, loss_cls: 0.0963, acc: 97.1973, loss_bbox: 0.0364, loss_mask: 0.1638, loss: 0.3077
2021-12-21 22:49:56,887 - mmdet - INFO - Epoch [11][750/930]    lr: 2.000e-03, eta: 0:04:33, time: 0.254, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0094, loss_rpn_bbox: 0.0036, loss_cls: 0.0973, acc: 97.2891, loss_bbox: 0.0380, loss_mask: 0.1839, loss: 0.3322
2021-12-21 22:50:09,618 - mmdet - INFO - Epoch [11][800/930]    lr: 2.000e-03, eta: 0:04:21, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0027, loss_cls: 0.0944, acc: 97.2871, loss_bbox: 0.0328, loss_mask: 0.1534, loss: 0.2929
2021-12-21 22:50:22,074 - mmdet - INFO - Epoch [11][850/930]    lr: 2.000e-03, eta: 0:04:09, time: 0.249, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0081, loss_rpn_bbox: 0.0032, loss_cls: 0.0865, acc: 97.5176, loss_bbox: 0.0299, loss_mask: 0.1504, loss: 0.2781
2021-12-21 22:50:35,180 - mmdet - INFO - Epoch [11][900/930]    lr: 2.000e-03, eta: 0:03:57, time: 0.262, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0074, loss_rpn_bbox: 0.0034, loss_cls: 0.0934, acc: 97.2832, loss_bbox: 0.0353, loss_mask: 0.1628, loss: 0.3022
2021-12-21 22:50:42,718 - mmdet - INFO - Saving checkpoint at 11 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.9 task/s, elapsed: 86s, ETA:     0s2021-12-21 22:52:10,927 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.79s).
Accumulating evaluation results...
DONE (t=0.39s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.033
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.306
2021-12-21 22:52:12,261 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.93s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.096
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.026
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.299
2021-12-21 22:52:13,960 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:52:13,960 - mmdet - INFO - Epoch(val) [11][764]   bbox_mAP: 0.0280, bbox_mAP_50: 0.0400, bbox_mAP_75: 0.0320, bbox_mAP_s: 0.3750, bbox_mAP_m: 0.0330, bbox_mAP_l: 0.0270, bbox_mAP_copypaste: 0.028 0.040 0.032 0.375 0.033 0.027, segm_mAP: 0.0280, segm_mAP_50: 0.0400, segm_mAP_75: 0.0340, segm_mAP_s: 0.0960, segm_mAP_m: 0.0320, segm_mAP_l: 0.0260, segm_mAP_copypaste: 0.028 0.040 0.034 0.096 0.032 0.026
/mnt/data01/home/gyf/projects/mmdetection/mmdet/core/mask/structures.py:1070: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  bitmap_mask = maskUtils.decode(rle).astype(np.bool)
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/mmcv/runner/hooks/logger/text.py:112: DeprecationWarning: an integer is required (got type float).  Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
  mem_mb = torch.tensor([mem / (1024 * 1024)],
```

**Epoch 12:**

```
2021-12-21 22:52:29,174 - mmdet - INFO - Epoch [12][50/930]     lr: 2.000e-04, eta: 0:03:36, time: 0.302, data_time: 0.053, memory: 3050, loss_rpn_cls: 0.0076, loss_rpn_bbox: 0.0029, loss_cls: 0.0989, acc: 97.1406, loss_bbox: 0.0379, loss_mask: 0.1692, loss: 0.3166
2021-12-21 22:52:41,662 - mmdet - INFO - Epoch [12][100/930]    lr: 2.000e-04, eta: 0:03:24, time: 0.250, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0107, loss_rpn_bbox: 0.0033, loss_cls: 0.0990, acc: 97.1719, loss_bbox: 0.0346, loss_mask: 0.1512, loss: 0.2988
2021-12-21 22:52:53,884 - mmdet - INFO - Epoch [12][150/930]    lr: 2.000e-04, eta: 0:03:12, time: 0.245, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0110, loss_rpn_bbox: 0.0032, loss_cls: 0.0933, acc: 97.4023, loss_bbox: 0.0318, loss_mask: 0.1617, loss: 0.3010
2021-12-21 22:53:06,489 - mmdet - INFO - Epoch [12][200/930]    lr: 2.000e-04, eta: 0:02:59, time: 0.252, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0078, loss_rpn_bbox: 0.0031, loss_cls: 0.0940, acc: 97.2930, loss_bbox: 0.0370, loss_mask: 0.1674, loss: 0.3094
2021-12-21 22:53:19,043 - mmdet - INFO - Epoch [12][250/930]    lr: 2.000e-04, eta: 0:02:47, time: 0.251, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0087, loss_rpn_bbox: 0.0031, loss_cls: 0.0910, acc: 97.3789, loss_bbox: 0.0323, loss_mask: 0.1516, loss: 0.2866
2021-12-21 22:53:31,721 - mmdet - INFO - Epoch [12][300/930]    lr: 2.000e-04, eta: 0:02:35, time: 0.254, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0063, loss_rpn_bbox: 0.0025, loss_cls: 0.0947, acc: 97.1328, loss_bbox: 0.0337, loss_mask: 0.1471, loss: 0.2842
2021-12-21 22:53:44,582 - mmdet - INFO - Epoch [12][350/930]    lr: 2.000e-04, eta: 0:02:23, time: 0.257, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0076, loss_rpn_bbox: 0.0030, loss_cls: 0.0960, acc: 97.2754, loss_bbox: 0.0382, loss_mask: 0.1741, loss: 0.3189
2021-12-21 22:53:57,471 - mmdet - INFO - Epoch [12][400/930]    lr: 2.000e-04, eta: 0:02:10, time: 0.258, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0073, loss_rpn_bbox: 0.0026, loss_cls: 0.0936, acc: 97.2344, loss_bbox: 0.0371, loss_mask: 0.1654, loss: 0.3061
2021-12-21 22:54:10,132 - mmdet - INFO - Epoch [12][450/930]    lr: 2.000e-04, eta: 0:01:58, time: 0.253, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0068, loss_rpn_bbox: 0.0029, loss_cls: 0.0916, acc: 97.3145, loss_bbox: 0.0337, loss_mask: 0.1552, loss: 0.2901
2021-12-21 22:54:22,944 - mmdet - INFO - Epoch [12][500/930]    lr: 2.000e-04, eta: 0:01:46, time: 0.256, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0082, loss_rpn_bbox: 0.0030, loss_cls: 0.0963, acc: 97.2285, loss_bbox: 0.0353, loss_mask: 0.1573, loss: 0.3003
2021-12-21 22:54:36,039 - mmdet - INFO - Epoch [12][550/930]    lr: 2.000e-04, eta: 0:01:33, time: 0.262, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0083, loss_rpn_bbox: 0.0033, loss_cls: 0.0970, acc: 97.1484, loss_bbox: 0.0365, loss_mask: 0.1605, loss: 0.3057
2021-12-21 22:54:48,906 - mmdet - INFO - Epoch [12][600/930]    lr: 2.000e-04, eta: 0:01:21, time: 0.258, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0089, loss_rpn_bbox: 0.0029, loss_cls: 0.0917, acc: 97.2832, loss_bbox: 0.0325, loss_mask: 0.1583, loss: 0.2943
2021-12-21 22:55:01,618 - mmdet - INFO - Epoch [12][650/930]    lr: 2.000e-04, eta: 0:01:09, time: 0.254, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0088, loss_rpn_bbox: 0.0028, loss_cls: 0.0936, acc: 97.2559, loss_bbox: 0.0377, loss_mask: 0.1702, loss: 0.3132
2021-12-21 22:55:14,380 - mmdet - INFO - Epoch [12][700/930]    lr: 2.000e-04, eta: 0:00:56, time: 0.255, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0078, loss_rpn_bbox: 0.0029, loss_cls: 0.0920, acc: 97.3027, loss_bbox: 0.0343, loss_mask: 0.1618, loss: 0.2989
2021-12-21 22:55:26,737 - mmdet - INFO - Epoch [12][750/930]    lr: 2.000e-04, eta: 0:00:44, time: 0.247, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0035, loss_cls: 0.0991, acc: 97.2461, loss_bbox: 0.0384, loss_mask: 0.1784, loss: 0.3295
2021-12-21 22:55:39,582 - mmdet - INFO - Epoch [12][800/930]    lr: 2.000e-04, eta: 0:00:32, time: 0.257, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0098, loss_rpn_bbox: 0.0029, loss_cls: 0.0934, acc: 97.3809, loss_bbox: 0.0344, loss_mask: 0.1634, loss: 0.3040
2021-12-21 22:55:51,856 - mmdet - INFO - Epoch [12][850/930]    lr: 2.000e-04, eta: 0:00:19, time: 0.246, data_time: 0.009, memory: 3050, loss_rpn_cls: 0.0090, loss_rpn_bbox: 0.0030, loss_cls: 0.0897, acc: 97.3887, loss_bbox: 0.0317, loss_mask: 0.1532, loss: 0.2865
2021-12-21 22:56:04,717 - mmdet - INFO - Epoch [12][900/930]    lr: 2.000e-04, eta: 0:00:07, time: 0.257, data_time: 0.008, memory: 3050, loss_rpn_cls: 0.0092, loss_rpn_bbox: 0.0027, loss_cls: 0.0912, acc: 97.3066, loss_bbox: 0.0355, loss_mask: 0.1518, loss: 0.2904
2021-12-21 22:56:12,185 - mmdet - INFO - Saving checkpoint at 12 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.9 task/s, elapsed: 86s, ETA:     0s2021-12-21 22:57:40,254 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.13s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.92s).
Accumulating evaluation results...
DONE (t=0.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.041
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.304
2021-12-21 22:57:41,738 - mmdet - INFO - Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.17s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=1.08s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.42s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.302
2021-12-21 22:57:43,654 - mmdet - INFO - Exp name: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py
2021-12-21 22:57:43,654 - mmdet - INFO - Epoch(val) [12][764]   bbox_mAP: 0.0290, bbox_mAP_50: 0.0410, bbox_mAP_75: 0.0340, bbox_mAP_s: 0.4000, bbox_mAP_m: 0.0340, bbox_mAP_l: 0.0280, bbox_mAP_copypaste: 0.029 0.041 0.034 0.400 0.034 0.028, segm_mAP: 0.0290, segm_mAP_50: 0.0400, segm_mAP_75: 0.0340, segm_mAP_s: 0.2370, segm_mAP_m: 0.0340, segm_mAP_l: 0.0280, segm_mAP_copypaste: 0.029 0.040 0.034 0.237 0.034 0.028
```

#### sockets bbox 测试

> 2021.12.21晚 706

**gyf@ubuntu ~/projects/mmdetection**

`% python tools/test.py configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py 		 work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets/latest.pth --eval bbox`

```
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
load checkpoint from local path: work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets/latest.pth
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.9 task/s, elapsed: 86s, ETA:     0s
Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.84s).
Accumulating evaluation results...
DONE (t=0.38s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.041
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.304
OrderedDict([('bbox_mAP', 0.029), ('bbox_mAP_50', 0.041), ('bbox_mAP_75', 0.034), ('bbox_mAP_s', 0.4), ('bbox_mAP_m', 0.034), ('bbox_mAP_l', 0.028), ('bbox_mAP_copypaste', '0.029 0.041 0.034 0.400 0.034 0.028')])
```

#### sockets bbox segm test

**gyf@ubuntu ~/projects/mmdetection**
` % python tools/test.py configs/fenghuo/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets/latest.pth --eval bbox segm` 

(在上述路径下执行命令)

```
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
load checkpoint from local path: work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_sockets/latest.pth
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 764/764, 8.9 task/s, elapsed: 86s, ETA:     0s
Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.87s).
Accumulating evaluation results...
DONE (t=0.39s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.041
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.304

Evaluating segm...
/mnt/data01/home/gyf/projects/mmdetection/mmdet/datasets/coco.py:450: UserWarning: The key "bbox" is deleted for more accurate mask AP of small/medium/large instances since v2.12.0. This does not change the overall mAP calculation.
  warnings.warn(
Loading and preparing results...
DONE (t=0.10s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.95s).
Accumulating evaluation results...
/home/gyf/envs/miniconda3/envs/mmlab/lib/python3.9/site-packages/pycocotools/cocoeval.py:378: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
DONE (t=0.42s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.375
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.302
OrderedDict([('bbox_mAP', 0.029), ('bbox_mAP_50', 0.041), ('bbox_mAP_75', 0.034), ('bbox_mAP_s', 0.4), ('bbox_mAP_m', 0.034), ('bbox_mAP_l', 0.028), ('bbox_mAP_copypaste', '0.029 0.041 0.034 0.400 0.034 0.028'), ('segm_mAP', 0.029), ('segm_mAP_50', 0.04), ('segm_mAP_75', 0.034), ('segm_mAP_s', 0.237), ('segm_mAP_m', 0.034), ('segm_mAP_l', 0.028), ('segm_mAP_copypaste', '0.029 0.040 0.034 0.237 0.034 0.028')])
```

