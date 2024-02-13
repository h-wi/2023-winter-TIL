# The new config inherits a base config to highlight the necessary modification
# _base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# _base_ = 'faster-rcnn_r50_fpn_2x_coco.py'

# _base_ = 'faster-rcnn_r101_fpn_2x_coco.py'

# _base_ = 'faster-rcnn_regnetx-3.2GF_fpn_2x_coco.py'

_base_ = 'faster-rcnn_hrnetv2p-w18-2x_coco.py'

model = dict(
    roi_head = dict(
        bbox_head=dict(num_classes=3)
    )
)
# Modify dataset related settings
# data_root = '../facenet_dataset'
data_root = ''
metainfo = {
    'classes': ('with_mask', 'without_mask', 'mask_weared_incorrect')
}
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='../facenet_dataset/annotations/train/annotation_coco_mask.json',
        data_prefix=dict(img='../facenet_dataset/images/train/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='../facenet_dataset/annotations/val/annotation_coco_mask.json',
        data_prefix=dict(img='../facenet_dataset/images/val/')))

test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file='../facenet_dataset/annotations/val/annotation_coco_mask.json')
test_evaluator = val_evaluator
