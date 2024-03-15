epochs_mapper = {
    "action_recognition": "train.num_epochs",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "bpnet": "num_epoch",
    "classification_pyt": "train.train_config.runner.max_epochs",
    "classification_tf1": "train_config.n_epochs",
    "classification_tf2": "train.num_epochs",
    "deformable_detr": "train.num_epochs",
    "detectnet_v2": "training_config.num_epochs",
    "dino": "train.num_epochs",
    "dssd": "training_config.num_epochs",
    "efficientdet_tf1": "training_config.num_epochs",
    "efficientdet_tf2": "train.num_epochs",
    "faster_rcnn": "training_config.num_epochs",
    "fpenet": "num_epoch",
    "lprnet": "training_config.num_epochs",
    "mal": "train.num_epochs",
    "mask_rcnn": "num_epochs",
    "ml_recog": "train.num_epochs",
    "multitask_classification": "training_config.num_epochs",
    "ocdnet": "train.num_epochs",
    "ocrnet": "train.num_epochs",
    "optical_inspection": "train.num_epochs",
    "pointpillars": "train.num_epochs",
    "pose_classification": "train.num_epochs",
    "re_identification": "train.num_epochs",
    "retinanet": "training_config.num_epochs",
    "segformer": "train.max_iters",
    "ssd": "training_config.num_epochs",
    "unet": "training_config.epochs",
    "yolo_v3": "training_config.num_epochs",
    "yolo_v4": "training_config.num_epochs",
    "yolo_v4_tiny": "training_config.num_epochs",
}

backbone_mapper = {
    "action_recognition": "model.backbone",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "bpnet": "model.backbone_attributes.architecture",
    "classification_pyt": "model.backbone.type",
    "classification_tf1": "model_config.arch",
    "classification_tf2": "model.backbone",
    "deformable_detr": "model.backbone",
    "detectnet_v2": "model_config.arch",
    "dino": "model.backbone",
    "dssd": "dssd_config.arch",
    "efficientdet_tf1": "model_config.model_name",
    "efficientdet_tf2": "model.name",
    "faster_rcnn": "model_config.arch",
    "fpenet": "",
    "lprnet": "lpr_config.arch",
    "mal": "model.arch",
    "mask_rcnn": "maskrcnn_config.arch",
    "ml_recog": "model.backbone",
    "multitask_classification": "model_config.arch",
    "ocdnet": "model.backbone",
    "ocrnet": "model.backbone",
    "optical_inspection": "model.backbone",
    "pointpillars": "model.backbone_2d.name",
    "pose_classification": "",
    "re_identification": "model.backbone",
    "retinanet": "retinanet_config.arch",
    "segformer": "model.backbone.type",
    "ssd": "ssd_config.arch",
    "unet": "model_config.arch",
    "yolo_v3": "yolov3_config.arch",
    "yolo_v4": "yolov4_config.arch",
    "yolo_v4_tiny": "yolov4_config.arch",
}

image_size_mapper = {
    "action_recognition": "model.input_height,model.input_width",
    "analytics": "",
    "annotations": "",
    "augmentation": "",
    "auto_label": "",
    "bpnet": "dataloader.image_config.image_dims.height,dataloader.image_config.image_dims.width",
    "classification_pyt": "",
    "classification_tf1": "model_config.input_image_size",
    "classification_tf2": "model.input_height,model.input_width",
    "deformable_detr": "",
    "detectnet_v2": "augmentation_config.preprocessing.output_image_height,augmentation_config.preprocessing.output_image_width",
    "dino": "",
    "dssd": "augmentation_config.output_height,augmentation_config.output_width",
    "efficientdet_tf1": "dataset_config.image_size",
    "efficientdet_tf2": "model.input_height,model.input_width",
    "faster_rcnn": "model_config.input_image_config.size_height_width.height,model_config.input_image_config.size_height_width.width",
    "fpenet": "dataloader.image_info.image.height,dataloader.image_info.image.width",
    "lprnet": "augmentation_config.output_height,augmentation_config.output_width",
    "mal": "",
    "mask_rcnn": "data_config.image_size",
    "ml_recog": "model.input_height,model.input_width",
    "multitask_classification": "model_config.input_image_size",
    "ocdnet": "",
    "ocrnet": "model.input_height,model.input_width",
    "optical_inspection": "dataset.output_shape",
    "pointpillars": "",
    "pose_classification": "",
    "re_identification": "model.input_height,model.input_width",
    "retinanet": "augmentation_config.output_height,augmentation_config.output_width",
    "segformer": "model.input_height,model.input_width",
    "ssd": "augmentation_config.output_height,augmentation_config.output_width",
    "unet": "model_config.model_input_height,model_config.model_input_width",
    "yolo_v3": "augmentation_config.output_height,augmentation_config.output_width",
    "yolo_v4": "augmentation_config.output_height,augmentation_config.output_width",
    "yolo_v4_tiny": "augmentation_config.output_height,augmentation_config.output_width",
}

# Include your network if it has spec fields to load full network as PTM and loading backbone portion alone
ptm_mapper = {
    "backbone": {
        "classification_pyt": "model.backbone.pretrained",
        "dino": "model.pretrained_backbone_path",
    },
    "end_to_end": {
        "classification_pyt": "model.init_cfg.checkpoint",
        "dino": "train.pretrained_model_path",
    }
}
