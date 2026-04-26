from __future__ import annotations

from argparse import Namespace

import pytest

from src.config import RAGConfig
from src.main import _resolve_runtime_config
from src.runtime_models import HardwareProfile, detect_hardware_profile


pytestmark = pytest.mark.unit


def test_detect_hardware_profile_honors_force_device_env(monkeypatch):
    monkeypatch.setenv("TOKENSMITH_FORCE_DEVICE", "gpu")
    detect_hardware_profile.cache_clear()

    profile = detect_hardware_profile()

    assert profile.gpu_available is True
    assert profile.backend == "forced"

    detect_hardware_profile.cache_clear()


def test_resolve_runtime_models_prefers_gpu_profile_when_gpu_available(monkeypatch):
    monkeypatch.setattr(
        "src.config.detect_hardware_profile",
        lambda: HardwareProfile(True, "vulkan", "AMD GPU"),
    )
    monkeypatch.setattr("src.config.gguf_path_available", lambda _path: True)

    cfg = RAGConfig(runtime_model_profile="auto")
    selection = cfg.resolve_runtime_models()

    assert selection["selected_profile"] == "gpu"
    assert selection["embed_model"] == cfg.gpu_embed_model
    assert selection["gen_model"] == cfg.gpu_gen_model


def test_resolve_runtime_models_falls_back_when_gpu_model_missing(monkeypatch):
    monkeypatch.setattr(
        "src.config.detect_hardware_profile",
        lambda: HardwareProfile(True, "vulkan", "AMD GPU"),
    )
    monkeypatch.setattr(
        "src.config.gguf_path_available",
        lambda path: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" not in path,
    )

    cfg = RAGConfig(runtime_model_profile="auto")
    selection = cfg.resolve_runtime_models()

    assert selection["selected_profile"] == "baseline"
    assert selection["gen_model"] == cfg.gen_model
    assert selection["gpu_model_missing"] is True


def test_resolve_runtime_config_applies_cli_generation_override(monkeypatch):
    monkeypatch.setattr(
        "src.config.detect_hardware_profile",
        lambda: HardwareProfile(False, "cpu", "CPU only"),
    )

    cfg = RAGConfig()
    args = Namespace(model_path="models/custom-chat.gguf")

    resolved_cfg, selection = _resolve_runtime_config(args, cfg)

    assert resolved_cfg.embed_model == cfg.embed_model
    assert resolved_cfg.gen_model == "models/custom-chat.gguf"
    assert selection["selected_profile"] == "baseline"
