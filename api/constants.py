# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants values"""

TAO_NETWORKS = set(["classification_tf2", "efficientdet_tf2",
                    "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "visual_changenet", "deformable_detr", "dino", "segformer",  # PYT CV MODELS
                    "annotations", "analytics", "augmentation", "auto_label", "image"])  # Data_Service tasks.
_OD_NETWORKS = set(["detectnet_v2", "efficientdet_tf2", "deformable_detr", "dino"])
_PURPOSE_BUILT_MODELS = set(["action_recognition", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pose_classification", "re_identification", "centerpose", "visual_changenet"])

_TF2_NETWORKS = set(["classification_tf2", "efficientdet_tf2"])
_PYT_TAO_NETWORKS = set(["action_recognition", "deformable_detr", "dino", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "segformer", "visual_changenet"])
_PYT_PLAYGROUND_NETWORKS = set(["classification_pyt"])
_PYT_CV_NETWORKS = _PYT_TAO_NETWORKS | _PYT_PLAYGROUND_NETWORKS
_DATA_SERVICES_ACTIONS = set(["annotation_format_convert", "generate", "augment", "analyze", "validate"])
_DATA_GENERATE_ACTIONS = set(["augment", "validate"])

VALID_DSTYPES = ("object_detection", "semantic_segmentation", "image_classification",
                 "instance_segmentation", "character_recognition",  # CV
                 "action_recognition", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "visual_changenet")  # PYT CV MODELS
TAO_NETWORKS = set(["classification_tf2", "efficientdet_tf2",
                    "action_recognition", "classification_pyt", "mal", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "centerpose", "visual_changenet", "deformable_detr", "dino", "segformer",  # PYT CV MODELS
                    "annotations", "analytics", "augmentation", "auto_label", "image"])  # Data_Service tasks.
MEDICAL_CUSTOM_ARCHITECT = ["medical_custom", "medical_classification", "medical_detection", "medical_segmentation"]
MEDICAL_NETWORK_ARCHITECT = ["medical_vista3d", "medical_vista2d", "medical_annotation", "medical_genai", "medical_maisi"] + MEDICAL_CUSTOM_ARCHITECT
MEDICAL_AUTOML_ARCHITECT = ["medical_automl", "medical_automl_generated"]
MONAI_NETWORKS = set(MEDICAL_NETWORK_ARCHITECT + MEDICAL_AUTOML_ARCHITECT)  # Data_Service tasks.
VALID_NETWORKS = TAO_NETWORKS | MONAI_NETWORKS
NO_SPEC_ACTIONS_MODEL = ("evaluate", "retrain", "inference", "inference_seq", "inference_trt")  # Actions with **optional** specs
NO_PTM_MODELS = set([])  # These networks don't have a pretrained model that can be downloaded from ngc model registry
_ITER_MODELS = set(["segformer"])  # These networks operate on iterations instead of epochs

BACKBONE_AND_FULL_MODEL_PTM_SUPPORTING_NETWORKS = set(["dino", "classification_pyt"])  # These networks have fields in their config file which has both backbone only loading weights as well as full architecture loading; ex: model.pretrained_backbone_path and train.pretrained_model_path in dino

AUTOML_DISABLED_NETWORKS = ["mal"]  # These networks can't support AutoML
NO_VAL_METRICS_DURING_TRAINING_NETWORKS = set(["unet"])  # These networks can't support writing validation metrics at regular intervals during training, only at end of training they run evaluation
MISSING_EPOCH_FORMAT_NETWORKS = set(["classification_pyt", "detectnet_v2", "pointpillars", "segformer", "unet"])  # These networks have the epoch/iter number not following a format; ex: 1.pth instead of 001.pth
STATUS_JSON_MISMATCH_WITH_CHECKPOINT_EPOCH = set(["pointpillars", "detectnet_v2"])  # status json epoch number is 1 less than epoch number generated in checkppoint file

MONAI_DATASET_DEFAULT_SPECS = {
    "next_image_strategy": "sequential",
    "cache_image_url": "",
    "cache_force": False,
    "notify_study_urls": [],
    "notify_image_urls": [],
    "notify_label_urls": [],
}

VALID_MODEL_DOWNLOAD_TYPE = ("medical_bundle", "tao")
CACHE_TIME_OUT = 60 * 60  # cache timeout period in second
LAST_ACCESS_TIME_OUT = 60  # last access timeout period in second

CONTINUOUS_STATUS_KEYS = ["cur_iter", "epoch", "max_epoch", "eta", "time_per_epoch", "time_per_iter", "key_metric"]

NETWORK_METRIC_MAPPING = {"action_recognition": "val_acc",
                          "centerpose": "val_3DIoU",
                          "classification_pyt": "accuracy_top-1",
                          "classification_tf2": "val_accuracy",
                          "deformable_detr": "val_mAP50",
                          "detectnet_v2": "mean average precision",
                          "dino": "val_mAP50",
                          "efficientdet_tf2": "AP50",
                          "mal": "mIoU",
                          "ml_recog": "val Precision at Rank 1",
                          "ocdnet": "hmean",
                          "ocrnet": "val_acc",
                          "optical_inspection": "val_acc",
                          "pointpillars": "loss",
                          "pose_classification": "val_acc",
                          "re_identification": "cmc_rank_1",
                          "segformer": "Mean IOU",
                          "unet": "loss",
                          "visual_changenet": "val_acc"}

NETWORK_CONTAINER_MAPPING = {"action_recognition": "TAO_PYTORCH",
                             "annotations": "TAO_DS",
                             "auto_label": "TAO_DS",
                             "analytics": "TAO_DS",
                             "augmentation": "TAO_DS",
                             "centerpose": "TAO_PYTORCH",
                             "classification_pyt": "TAO_PYTORCH",
                             "classification_tf2": "TAO_TF2",
                             "deformable_detr": "TAO_PYTORCH",
                             "detectnet_v2": "TAO_TF2",
                             "dino": "TAO_PYTORCH",
                             "efficientdet_tf2": "TAO_TF2",
                             "image": "TAO_DS",
                             "mal": "TAO_PYTORCH",
                             "ml_recog": "TAO_PYTORCH",
                             "ocdnet": "TAO_PYTORCH",
                             "ocrnet": "TAO_PYTORCH",
                             "optical_inspection": "TAO_PYTORCH",
                             "pointpillars": "TAO_PYTORCH",
                             "pose_classification": "TAO_PYTORCH",
                             "re_identification": "TAO_PYTORCH",
                             "segformer": "TAO_PYTORCH",
                             "unet": "TAO_TF2",
                             "visual_changenet": "TAO_PYTORCH"}
