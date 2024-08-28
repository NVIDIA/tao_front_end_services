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

"""Defining enums for dataset and model formats and types"""
import enum


class DatasetType(str, enum.Enum):
    """Class defining dataset types in enum"""

    object_detection = 'object_detection'
    semantic_segmentation = 'semantic_segmentation'
    image_classification = 'image_classification'
    instance_segmentation = 'instance_segmentation'
    character_recognition = 'character_recognition'
    action_recognition = 'action_recognition'
    pointpillars = 'pointpillars'
    pose_classification = 'pose_classification'
    ml_recog = 'ml_recog'
    ocdnet = 'ocdnet'
    ocrnet = 'ocrnet'
    optical_inspection = 'optical_inspection'
    re_identification = 're_identification'
    visual_changenet = 'visual_changenet'
    centerpose = 'centerpose'
    not_restricted = 'not_restricted'
    user_custom = 'user_custom'


class DatasetFormat(str, enum.Enum):
    """Class defining dataset formats in enum"""

    kitti = 'kitti'
    pascal_voc = 'pascal_voc'
    raw = 'raw'
    coco_raw = 'coco_raw'
    unet = 'unet'
    coco = 'coco'
    lprnet = 'lprnet'
    train = 'train'
    test = 'test'
    default = 'default'
    custom = 'custom'
    classification_pyt = 'classification_pyt'
    classification_tf2 = 'classification_tf2'
    visual_changenet_segment = 'visual_changenet_segment'
    visual_changenet_classify = 'visual_changenet_classify'
    medical = 'medical'


class ExperimentNetworkArch(str, enum.Enum):
    """Class defining network types in enum"""

    # Tf networks
    detectnet_v2 = 'detectnet_v2'
    unet = 'unet'
    classification_tf2 = 'classification_tf2'
    efficientdet_tf2 = 'efficientdet_tf2'
    # PyT CV networks
    action_recognition = 'action_recognition'
    classification_pyt = 'classification_pyt'
    mal = 'mal'
    ml_recog = 'ml_recog'
    ocdnet = 'ocdnet'
    ocrnet = 'ocrnet'
    optical_inspection = 'optical_inspection'
    pointpillars = 'pointpillars'
    pose_classification = 'pose_classification'
    re_identification = 're_identification'
    deformable_detr = 'deformable_detr'
    dino = 'dino'
    segformer = 'segformer'
    visual_changenet = 'visual_changenet'
    centerpose = 'centerpose'
    # Data analytics networks
    auto_label = "auto_label"
    image = "image"
    # Monai Networks
    medical_vista3d = "medical_vista3d"
    medical_vista2d = "medical_vista2d"
    medical_segmentation = "medical_segmentation"
    medical_annotation = "medical_annotation"
    medical_classification = "medical_classification"
    medical_detection = "medical_detection"
    medical_automl = "medical_automl"
    medical_custom = "medical_custom"
    medical_genai = "medical_genai"
    medical_maisi = "medical_maisi"
    medical_automl_generated = "medical_automl_generated"


class Metrics(str, enum.Enum):
    """Class defining metric types in enum"""

    three_d_mAP = '3d mAP'
    AP = 'AP'
    AP50 = 'AP50'
    AP75 = 'AP75'
    APl = 'APl'
    APm = 'APm'
    APs = 'APs'
    ARl = 'ARl'
    ARm = 'ARm'
    ARmax1 = 'ARmax1'
    ARmax10 = 'ARmax10'
    ARmax100 = 'ARmax100'
    ARs = 'ARs'
    Hmean = 'Hmean'
    Mean_IOU = 'Mean IOU'
    Precision = 'Precision'
    Recall = 'Recall'
    Thresh = 'Thresh'
    accuracy = 'accuracy'
    m_accuracy = 'm_accuracy'
    avg_accuracy = 'avg_accuracy'
    accuracy_top_1 = 'accuracy_top-1'
    bev_mAP = 'bev mAP'
    cmc_rank_1 = 'cmc_rank_1'
    cmc_rank_10 = 'cmc_rank_10'
    cmc_rank_5 = 'cmc_rank_5'
    defect_acc = 'defect_acc'
    embedder_base_lr = 'embedder_base_lr'
    hmean = 'hmean'
    learning_rate = 'learning_rate'
    loss = 'loss'
    lr = 'lr'
    mAP = 'mAP'
    mAcc = 'mAcc'
    mIoU = 'mIoU'
    mIoU_large = 'mIoU_large'
    mIoU_medium = 'mIoU_medium'
    mIoU_small = 'mIoU_small'
    param_count = 'param_count'
    precision = 'precision'
    pruning_ratio = 'pruning_ratio'
    recall = 'recall'
    recall_rcnn_0_3 = 'recall/rcnn_0.3'
    recall_rcnn_0_5 = 'recall/rcnn_0.5'
    recall_rcnn_0_7 = 'recall/rcnn_0.7'
    recall_roi_0_3 = 'recall/roi_0.3'
    recall_roi_0_5 = 'recall/roi_0.5'
    recall_roi_0_7 = 'recall/roi_0.7'
    size = 'size'
    test_Mean_Average_Precision = 'test Mean Average Precision'
    test_Mean_Reciprocal_Rank = 'test Mean Reciprocal Rank'
    test_Precision_at_Rank_1 = 'test Precision at Rank 1'
    test_r_Precision = 'test r-Precision'
    test_AMI = 'test_AMI'
    test_NMI = 'test_NMI'
    test_acc = 'test_acc'
    test_fnr = 'test_fnr'
    test_fpr = 'test_fpr'
    test_mAP = 'test_mAP'
    test_mAP50 = 'test_mAP50'
    test_mf1 = 'test_mf1'
    test_miou = 'test_miou'
    test_mprecision = 'test_mprecision'
    test_mrecall = 'test_mrecall'
    top_k = 'top_k'
    train_acc = 'train_acc'
    train_accuracy = 'train_accuracy'
    train_fpr = 'train_fpr'
    train_loss = 'train_loss'
    trunk_base_lr = 'trunk_base_lr'
    val_Mean_Average_Precision = 'val Mean Average Precision'
    val_Mean_Reciprocal_Rank = 'val Mean Reciprocal Rank'
    val_Precision_at_Rank_1 = 'val Precision at Rank 1'
    val_r_Precision = 'val r-Precision'
    val_2DMPE = 'val_2DMPE'
    val_3DIoU = 'val_3DIoU'
    test_2DMPE = 'test_2DMPE'
    test_3DIoU = 'test_3DIoU'
    val_AMI = 'val_AMI'
    val_NMI = 'val_NMI'
    val_acc = 'val_acc'
    val_accuracy = 'val_accuracy'
    val_fpr = 'val_fpr'
    val_loss = 'val_loss'
    val_mAP = 'val_mAP'
    val_mAP50 = 'val_mAP50'
    val_mf1 = 'val_mf1'
    val_miou = 'val_miou'
    val_mprecision = 'val_mprecision'
    val_mrecall = 'val_mrecall'

    num_objects = 'num_objects'
