# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the API schema and runtime parity for SSL refinement."""

from dataclasses import asdict, fields

import pytest

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.config.dinov3.default_config import DINOv3DatasetConfig, ExperimentConfig
from nvidia_tao_core.config.dinov3.grit_score import GRITScoreConfig


def test_refinement_fields_are_exposed_in_api_schema():
    """Generated schemas expose manifest training and model-owned GRIT scoring."""
    schema = create_json_schema(dataclass_to_json(ExperimentConfig()))["properties"]
    manifest = schema["dataset"]["properties"]["train_manifest"]
    assert manifest["type"] == "string"
    assert "storage_type" in manifest["description"]
    scoring = schema["grit_score"]["properties"]
    assert scoring["neighbor_backend"]["enum"] == ["auto", "torch_exact", "faiss_exact"]
    assert scoring["batch_size"]["default"] == 12
    assert scoring["precomputed_consensus"]["default"] is False
    assert set(scoring) == {field.name for field in fields(GRITScoreConfig)}


def test_refinement_schema_matches_runtime_when_available():
    """Keep core defaults and metadata aligned with the optional PyTorch runtime."""
    runtime = pytest.importorskip("nvidia_tao_pytorch.config.dinov3.default_config")
    assert hasattr(runtime, "GRITScoreConfig"), "Runtime release lacks the DEFT scoring schema"
    assert asdict(GRITScoreConfig()) == asdict(runtime.GRITScoreConfig())
    core_fields = {field.name: dict(field.metadata) for field in fields(GRITScoreConfig)}
    runtime_fields = {field.name: dict(field.metadata) for field in fields(runtime.GRITScoreConfig)}
    assert core_fields == runtime_fields
    core_manifest = DINOv3DatasetConfig.__dataclass_fields__["train_manifest"]
    runtime_manifest = runtime.DINOv3DatasetConfig.__dataclass_fields__["train_manifest"]
    assert core_manifest.default == runtime_manifest.default
    assert dict(core_manifest.metadata) == dict(runtime_manifest.metadata)
