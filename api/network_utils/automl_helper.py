automl_list_helper = {
    "yolo_v3": {
        "list_1_backbone": {
            "yolov3_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
            },
        },
        "list_1_normal": {},
    },
    "yolo_v4": {
        "list_1_backbone": {
            "yolov4_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
            },
        },
        "list_1_normal": {},
    },
    "yolo_v4_tiny": {
        "list_1_backbone": {
            "yolov4_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
            },
        },
        "list_1_normal": {},
    },
    "deformable_detr": {
        "list_2": {
            "dataset.augmentation.scales": ("img_size", "base_parameter"),
            "train.optim.lr_steps": ("lr_steps", "train.num_epochs"),
        }
    },
    "ml_recog": {
        "list_3": {
            "dataset.gaussian_blur.kernel": ("img_size", "model.input_height"),
        }
    },
    "centerpose": {
        "list_2": {
            "train.optim.lr_steps": ("lr_steps", "train.num_epochs")
        }
    },
    "ocrnet": {
        "list_2": {
            "dataset.augmentation.gaussian_radius_list": ("img_size", "model.input_height"),
        }
    },
    "action_recognition": {
        "list_2": {
            "train.optim.lr_steps": ("lr_steps", "train.num_epochs"),
        }
    },
    "lprnet": {
        "list_2": {
            "augmentation_config.gaussian_kernel_size": ("img_size", "augmentation_config.output_height"),
        }
    },
    "pointpillars": {
        "list_2": {
            "train.decay_step_list": ("lr_steps", "train.num_epochs"),
        }
    },
    "pose_classification": {
        "list_2": {
            "train.optim.lr_steps": ("lr_steps", "train.num_epochs"),
        }
    },
    "re_identification": {
        "list_2": {
            "train.optim.lr_steps": ("lr_steps", "train.num_epochs"),
        }
    },
    "faster_rcnn": {
        "list_1_backbone": {
            "model_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
                "efficientnet": (0, 7),
            },
        },
        "list_1_normal": {},
    },
    "detectnet_v2": {
        "list_1_backbone": {
            "model_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
                "efficientnet": (0, 7),
            },
        },
        "list_1_normal": {},
    },
    "classification_tf2": {
        "list_1_backbone": {
            "model.freeze_blocks": {
                "resnet_10": (0, 3),
                "resnet_18": (0, 3),
                "resnet_34": (0, 3),
                "resnet_50": (0, 3),
                "resnet_101": (0, 3),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "efficientnet-b0": (0, 7),
                "efficientnet-b1": (0, 7),
                "efficientnet-b2": (0, 7),
                "efficientnet-b3": (0, 7),
                "efficientnet-b4": (0, 7),
                "efficientnet-b5": (0, 7),
                "efficientnet-b6": (0, 7),
                "efficientnet-b7": (0, 7),
            },
        },
    },
    "efficientdet_tf2": {
        "list_1_backbone": {
            "model.freeze_blocks": {
                "efficientdet-d0": (0, 7),
                "efficientdet-d1": (0, 7),
                "efficientdet-d2": (0, 7),
                "efficientdet-d3": (0, 7),
                "efficientdet-d4": (0, 7),
                "efficientdet-d5": (0, 7),
            },
        },
    },
    "mal": {
        "list_1_backbone": {
            "model.frozen_stages": {
                "vit-deit-tiny/16": (-1, 12),
                "vit-deit-small/16": (-1, 12),
                "vit-mae-base/16": (-1, 12),
                'vit-mae-large/16': (-1, 12),
                'vit-mae-huge/14': (-1, 12),
                "fan_tiny_12_p16_224": (-1, 8),
                "fan_small_12_p16_224": (-1, 8),
                "fan_base_18_p16_224": (-1, 8),
                "fan_large_24_p16_224": (-1, 8),
                "fan_tiny_8_p4_hybrid": (-1, 8),
                "fan_small_12_p4_hybrid": (-1, 8),
                "fan_base_16_p4_hybrid": (-1, 8),
                "fan_large_16_p4_hybrid": (-1, 8)
            },
        },
    },
    "ssd": {
        "list_1_backbone": {
            "ssd_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
                "efficientnet_b0": (0, 7),
                "squeezenet": (0, 8)
            },
        },
        "list_1_normal": {},
    },
    "dssd": {
        "list_1_backbone": {
            "dssd_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
                "efficientnet_b0": (0, 7),
                "squeezenet": (0, 8)
            },
        },
        "list_1_normal": {},
    },
    "retinanet": {
        "list_1_backbone": {
            "retinanet_config.freeze_blocks": {
                "resnet": (0, 3),
                "vgg": (1, 5),
                "googlenet": (0, 7),
                "mobilenet_v1": (0, 11),
                "mobilenet_v2": (0, 13),
                "darknet": (0, 5),
                "efficientnet_b0": (0, 7),
                "squeezenet": (0, 8)
            },
        },
        "list_1_normal": {},
    },
}
