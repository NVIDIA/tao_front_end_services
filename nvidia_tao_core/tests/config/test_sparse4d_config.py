# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Sparse4D mixed 2D/3D service schema."""

import re

from omegaconf import OmegaConf

from nvidia_tao_core.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_core.microservices.utils.core_utils import (
    get_microservices_network_and_action,
    read_network_config,
)
from nvidia_tao_core.scripts.generate_schema import generate_schema


DATASET_COTRAIN_DEFAULTS = {
    "ltt_2dgt_sidecar_dir": "",
    "ltt_2dgt_frame_regex": "",
    "ltt_2dgt_dedup_regex": r"^CT[\w.]+?__",
    "ltt_2dgt_cache_size": 8,
    "ltt_2dgt_warn_on_miss": True,
    "rtdetr_2d_cache_dir": "",
    "rtdetr_2d_cache_path": "",
    "rtdetr_2d_dedup_regex": r"^CT[\w.]+?__",
    "rtdetr_2d_score_thr": 0.0,
    "rtdetr_2d_per_class_score_thr": {},
    "rtdetr_2d_cache_size": 4,
    "rtdetr_2d_mark_real": True,
    "resize_to_canonical_2d": False,
    "canonical_2d_height": 1080,
    "canonical_2d_width": 1920,
    "sync_route": False,
    "real_scene_keywords": [],
    "real_block_prob": -1.0,
    "scene_switch_iters": 0,
}

LOOSE_TO_TIGHT_DEFAULTS = {
    "enable": False,
    "mlp_ckpt": "",
    "loss_weight": 0.1,
    "num_classes": 0,
    "tight_l1_weight": 1.0,
    "containment_weight": 1.0,
    "box2d_key": "gt_boxes_2d_visible",
    "occ_key": "gt_occ_weight",
    "instance_id_key": "instance_id",
    "ego2cam_key": "cam2world_transform",
    "min_gt_area": 1.0,
    "eps": 0.1,
    "pseudo_enable": False,
    "det_box_key": "det_boxes_2d",
    "det_cls_key": "det_classes_2d",
    "det_score_key": "det_scores_2d",
    "has_3d_gt_key": "has_3d_gt",
    "giou_thr": 0.3,
    "cost_giou": 2.0,
    "cost_l1": 1.0,
    "cost_cls": 1.0,
    "det_score_thr": 0.0,
    "min_cams": 1,
    "dedup_dist": 0.0,
    "class_gate": True,
    "pseudo_box_weight": 0.1,
    "pseudo_cls_weight": 1.0,
    "sv_depth_weight": 0.0,
    "sv_size_weight": 0.25,
    "sv_yaw_weight": 0.0,
}

SV_AUX_DEFAULTS = {
    "enable": False,
    "in_channels": 256,
    "num_classes": 0,
    "roi_size": 7,
    "hidden_dim": 256,
    "fpn_strides": [4, 8, 16, 32],
    "use_level": 1,
    "loss_weight": 1.0,
    "det_box_key": "det_boxes_2d",
    "det_cls_key": "det_classes_2d",
    "min_box_size": 2.0,
}

COTRAIN_METRICS = (
    "loss_box_2d_0",
    "loss_box_2d_pseudo_5",
    "loss_cls_pseudo_3",
    "loss_sv_head_1",
    "loss_sv_aux_cls",
    "loss_param_touch",
)

COTRAIN_METRIC_PATTERNS = {
    r"^loss_box_2d_\d+$",
    r"^loss_box_2d_pseudo_\d+$",
    r"^loss_cls_pseudo_\d+$",
    r"^loss_sv_head_\d+$",
    r"^loss_sv_aux_cls$",
    r"^loss_param_touch$",
}


def _select(values, keys):
    """Return the requested keys from a generated default dictionary."""
    return {key: values[key] for key in keys}


def test_sparse4d_train_schema_exposes_cotrain_defaults():
    """Generated train schema mirrors the runtime's opt-in co-training surface."""
    schema = generate_schema("sparse4d", "train")
    defaults = schema["default"]

    dataset = defaults["dataset"]
    assert _select(dataset, DATASET_COTRAIN_DEFAULTS) == DATASET_COTRAIN_DEFAULTS
    assert dataset["eval_dist_fcn"] == "center_distance"
    assert dataset["eval_hota"] is False

    model = defaults["model"]
    assert model["head"]["loose_to_tight"] == LOOSE_TO_TIGHT_DEFAULTS
    assert model["head"]["instance_bank"]["reset_on_time_gap"] is False
    assert model["cotrain_param_touch"] is False
    assert model["sv_scene_keywords"] == ["SV2D"]
    assert model["sv_aux_head"] == SV_AUX_DEFAULTS
    assert defaults["train"]["scrub_nan_gradients"] is False

    dataset_schema = schema["properties"]["dataset"]["properties"]
    assert dataset_schema["eval_dist_fcn"]["enum"] == [
        "center_distance",
        "iou_3d",
        "both",
    ]
    assert dataset_schema["rtdetr_2d_score_thr"]["minimum"] == 0.0
    assert dataset_schema["rtdetr_2d_score_thr"]["maximum"] == 1.0
    assert dataset_schema["real_block_prob"]["minimum"] == -1.0
    assert dataset_schema["real_block_prob"]["maximum"] == 1.0


def test_sparse4d_structured_config_accepts_cotrain_overrides():
    """OmegaConf retains nested route settings instead of dropping service input."""
    config = OmegaConf.merge(
        OmegaConf.structured(ExperimentConfig()),
        {
            "dataset": {
                "ltt_2dgt_sidecar_dir": "/results/ltt_2dgt",
                "rtdetr_2d_per_class_score_thr": {"person": 0.75},
                "sync_route": True,
                "real_scene_keywords": ["SV2D", "real"],
            },
            "model": {
                "cotrain_param_touch": True,
                "head": {
                    "instance_bank": {"reset_on_time_gap": True},
                    "loose_to_tight": {
                        "enable": True,
                        "mlp_ckpt": "/results/loose_to_tight_mlp.pth",
                    },
                },
                "sv_aux_head": {"enable": True},
            },
            "train": {"scrub_nan_gradients": True},
        },
    )

    assert config.dataset.ltt_2dgt_sidecar_dir == "/results/ltt_2dgt"
    assert config.dataset.rtdetr_2d_per_class_score_thr == {"person": 0.75}
    assert config.dataset.sync_route is True
    assert config.model.head.instance_bank.reset_on_time_gap is True
    assert config.model.head.loose_to_tight.enable is True
    assert config.model.sv_aux_head.enable is True
    assert config.train.scrub_nan_gradients is True


def test_sparse4d_inference_schema_exposes_nvschema_fps():
    """Inference clients can control NVSchema timing without changing old output."""
    schema = generate_schema("sparse4d", "inference")
    inference = schema["properties"]["inference"]["properties"]

    assert schema["default"]["inference"]["nvschema_fps"] == 0.0
    assert inference["nvschema_fps"]["minimum"] == 0.0


def test_sparse4d_service_accepts_cotrain_metrics():
    """The service metric allowlist recognizes every route-specific loss key."""
    patterns = read_network_config("sparse4d")["metrics"]["dynamic_metric_patterns"]

    assert COTRAIN_METRIC_PATTERNS <= set(patterns)
    for metric in COTRAIN_METRICS:
        assert any(re.fullmatch(pattern, metric) for pattern in patterns), metric


def test_sparse4d_service_data_source_paths_exist_in_schema():
    """Every Sparse4D service override still resolves through generated schemas."""
    config = read_network_config("sparse4d")

    for action, overrides in config["data_sources"].items():
        network, mapped_action = get_microservices_network_and_action(
            "sparse4d", action
        )
        for override in overrides:
            default = generate_schema(network, mapped_action)["default"]
            for key in override.split("."):
                assert key in default, (
                    f"Override path {override} not found in "
                    f"{network} {mapped_action} schema"
                )
                default = default[key]
