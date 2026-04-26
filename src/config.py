from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List
import re

import yaml
import pathlib

from src.preprocessing.chunking import (
    ChunkStrategy,
    SectionRecursiveStrategy,
    SectionRecursiveConfig,
    NaiveRecursiveStrategy,
    NaiveRecursiveConfig,
    ChunkConfig,
)
from src.runtime_models import (
    AUTO_MODEL_PROFILE,
    BASELINE_MODEL_PROFILE,
    GPU_MODEL_PROFILE,
    SUPPORTED_MODEL_PROFILES,
    detect_hardware_profile,
    gguf_path_available,
)


SUPPORTED_RETRIEVERS = {"faiss", "bm25", "index_keywords", "graph"}

@dataclass
class RAGConfig:
    # chunking
    chunk_config: ChunkConfig = field(init=False)
    chunk_mode: str = "recursive_sections"
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # retrieval + ranking
    top_k: int = 10
    num_candidates: int = 60
    embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    gpu_embed_model: str = "models/Qwen3-Embedding-4B-Q8_0.gguf"
    ensemble_method: str = "rrf"
    rrf_k: int  = 60
    ranker_weights: Dict[str, float] = field(
        default_factory=lambda: {"faiss": 0.55, "bm25": 0.2, "index_keywords": 0.1, "graph": 0.15}
    )
    enabled_retrievers: List[str] | None = None
    rerank_mode: str = ""
    rerank_top_k: int = 5

    # generation
    max_gen_tokens: int = 400
    gen_model: str = "models/qwen2.5-3b-instruct-q8_0.gguf"
    gpu_gen_model: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    runtime_model_profile: str = AUTO_MODEL_PROFILE
    
    # testing
    system_prompt_mode: str = "baseline"
    disable_chunks: bool = False
    use_golden_chunks: bool = False
    output_mode: str = "terminal"
    metrics: list = field(default_factory=lambda: ["all"])

    # query enhancement
    use_hyde: bool = False
    hyde_max_tokens: int = 300
    use_double_prompt: bool = False

    # conversational memory
    enable_history: bool = True
    max_history_turns: int = 3
    
    # index parameters
    use_indexed_chunks: bool = False
    extracted_index_path: os.PathLike = "data/extracted_index.json"
    page_to_chunk_map_path: os.PathLike = "index/sections/textbook_index_page_to_chunk_map.json"
    graph_artifact_name: str = "graph.json"
    graph_max_entities_per_chunk: int = 24

    # user feedback modeling
    enable_topic_extraction: bool = False

    # ---------- factory + validation ----------
    @classmethod
    def from_yaml(cls, path: os.PathLike) -> RAGConfig:
        with open(path, 'r') as f:
            data = yaml.safe_load(open(path))
        return cls(**data)
    
    def __post_init__(self):
        """Validation logic runs automatically after initialization."""
        assert self.top_k > 0, "top_k must be > 0"
        assert self.num_candidates >= self.top_k, "num_candidates must be >= top_k"
        assert self.ensemble_method.lower() in {"linear","weighted","rrf"}
        if self.runtime_model_profile not in SUPPORTED_MODEL_PROFILES:
            raise ValueError(
                "runtime_model_profile must be one of "
                f"{sorted(SUPPORTED_MODEL_PROFILES)}"
            )
        unknown = set(self.ranker_weights) - SUPPORTED_RETRIEVERS
        if unknown:
            raise ValueError(f"Unsupported retrievers in ranker_weights: {sorted(unknown)}")
        if self.enabled_retrievers is not None:
            invalid = set(self.enabled_retrievers) - SUPPORTED_RETRIEVERS
            if invalid:
                raise ValueError(f"Unsupported enabled_retrievers: {sorted(invalid)}")
        self.ranker_weights = {
            name: float(self.ranker_weights.get(name, 0.0))
            for name in sorted(SUPPORTED_RETRIEVERS)
        }
        self.chunk_config = self.get_chunk_config()
        self.chunk_config.validate()

    # ---------- chunking + artifact name helpers ----------

    def get_chunk_config(self) -> ChunkConfig:
        """Parse chunk configuration from YAML."""
        if self.chunk_mode == "recursive_sections":
            return SectionRecursiveConfig(
                recursive_chunk_size=self.chunk_size,
                recursive_overlap=self.chunk_overlap
            )
        if self.chunk_mode == "recursive_naive":
            return NaiveRecursiveConfig(
                recursive_chunk_size=self.chunk_size,
                recursive_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(
                f"Unknown chunk_mode: {self.chunk_mode}. Supported: recursive_sections, recursive_naive"
            )

    def get_chunk_strategy(self) -> ChunkStrategy:
        if isinstance(self.chunk_config, SectionRecursiveConfig):
            return SectionRecursiveStrategy(self.chunk_config)
        if isinstance(self.chunk_config, NaiveRecursiveConfig):
            return NaiveRecursiveStrategy(self.chunk_config)
        raise ValueError(f"Unknown chunk config type: {self.chunk_config.__class__.__name__}")

    def get_artifacts_root_directory(self) -> pathlib.Path:
        root_dir = pathlib.Path("index")
        root_dir.mkdir(parents=True, exist_ok=True)
        return root_dir

    def get_artifacts_directory(self) -> os.PathLike:
        """Legacy chunking-specific artifact directory."""
        strategy = self.get_chunk_strategy()
        strategy_dir = self.get_artifacts_root_directory() / strategy.artifact_folder_name()
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir

    @staticmethod
    def _sanitize_artifact_component(value: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return sanitized or "document"

    def get_chunking_directory_label(self) -> str:
        if isinstance(self.chunk_config, SectionRecursiveConfig):
            return "structure"
        if isinstance(self.chunk_config, NaiveRecursiveConfig):
            return "recursive"
        return "custom"

    def build_versioned_artifacts_directory(self, source_document: os.PathLike) -> pathlib.Path:
        source_name = pathlib.Path(source_document).stem
        dir_stem = (
            f"{self._sanitize_artifact_component(source_name)}_"
            f"{self.get_chunking_directory_label()}"
        )
        root_dir = self.get_artifacts_root_directory()
        version = 1
        while True:
            candidate = root_dir / f"{dir_stem}_{version}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            version += 1

    def discover_artifact_directories(self) -> List[pathlib.Path]:
        root_dir = self.get_artifacts_root_directory()
        candidates: List[pathlib.Path] = []

        legacy_dir = pathlib.Path(self.get_artifacts_directory())
        if legacy_dir.exists():
            candidates.append(legacy_dir)

        candidates.extend(
            sorted(
                [path for path in root_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            )
        )

        deduped: List[pathlib.Path] = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(candidate)
        return deduped

    def resolve_artifacts_directory(self, index_prefix: str) -> pathlib.Path:
        for candidate_dir in self.discover_artifact_directories():
            if (candidate_dir / f"{index_prefix}_chunks.pkl").exists():
                return candidate_dir
        fallback_dir = pathlib.Path(self.get_artifacts_directory())
        if fallback_dir.exists():
            return fallback_dir
        raise FileNotFoundError(
            f"Could not find artifacts for index_prefix '{index_prefix}' under '{self.get_artifacts_root_directory()}'."
        )

    def get_page_to_chunk_map_artifact_path(
        self,
        index_prefix: str,
        *,
        artifacts_dir: os.PathLike | None = None,
    ) -> pathlib.Path:
        base_dir = pathlib.Path(artifacts_dir) if artifacts_dir is not None else self.resolve_artifacts_directory(index_prefix)
        return base_dir / f"{index_prefix}_page_to_chunk_map.json"

    def get_graph_artifact_path(
        self,
        index_prefix: str,
        *,
        artifacts_dir: os.PathLike | None = None,
    ) -> pathlib.Path:
        artifact_name = self.graph_artifact_name
        if "{index_prefix}" in artifact_name:
            artifact_name = artifact_name.format(index_prefix=index_prefix)
        elif artifact_name == "graph.json":
            artifact_name = f"{index_prefix}_graph.json"
        base_dir = pathlib.Path(artifacts_dir) if artifacts_dir is not None else self.resolve_artifacts_directory(index_prefix)
        return base_dir / artifact_name

    def get_ingestion_metadata_path(
        self,
        index_prefix: str,
        *,
        artifacts_dir: os.PathLike | None = None,
    ) -> pathlib.Path:
        base_dir = pathlib.Path(artifacts_dir) if artifacts_dir is not None else self.resolve_artifacts_directory(index_prefix)
        return base_dir / f"{index_prefix}_ingestion_metadata.json"

    def get_tuning_directory(
        self,
        index_prefix: str,
        *,
        artifacts_dir: os.PathLike | None = None,
    ) -> pathlib.Path:
        base_dir = pathlib.Path(artifacts_dir) if artifacts_dir is not None else self.resolve_artifacts_directory(index_prefix)
        tuning_dir = base_dir / f"{index_prefix}_tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        return tuning_dir

    def get_tuning_examples_path(self, index_prefix: str, *, artifacts_dir: os.PathLike | None = None) -> pathlib.Path:
        return self.get_tuning_directory(index_prefix, artifacts_dir=artifacts_dir) / "demonstrations.jsonl"

    def get_tuning_state_path(self, index_prefix: str, *, artifacts_dir: os.PathLike | None = None) -> pathlib.Path:
        return self.get_tuning_directory(index_prefix, artifacts_dir=artifacts_dir) / "learned_weights.json"

    def get_tuning_manifest_path(self, index_prefix: str, *, artifacts_dir: os.PathLike | None = None) -> pathlib.Path:
        return self.get_tuning_directory(index_prefix, artifacts_dir=artifacts_dir) / "manifest.json"

    def get_tuning_export_path(self, index_prefix: str, *, artifacts_dir: os.PathLike | None = None) -> pathlib.Path:
        return self.get_tuning_directory(index_prefix, artifacts_dir=artifacts_dir) / "ranker_weights.export.yaml"

    def get_tuning_dataset_path(self, index_prefix: str, *, artifacts_dir: os.PathLike | None = None) -> pathlib.Path:
        return self.get_tuning_directory(index_prefix, artifacts_dir=artifacts_dir) / "prompt_chunk_dataset.jsonl"

    def get_tuning_benchmark_directory(self, index_prefix: str, *, artifacts_dir: os.PathLike | None = None) -> pathlib.Path:
        benchmark_dir = self.get_tuning_directory(index_prefix, artifacts_dir=artifacts_dir) / "benchmarks"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        return benchmark_dir

    def get_enabled_retrievers(self) -> List[str]:
        if self.enabled_retrievers is not None:
            return [name for name in self.enabled_retrievers if name in SUPPORTED_RETRIEVERS]
        return [name for name, weight in self.ranker_weights.items() if weight > 0]

    def get_active_ranker_weights(self) -> Dict[str, float]:
        enabled = self.get_enabled_retrievers()
        active = {
            name: self.ranker_weights.get(name, 0.0)
            for name in enabled
            if self.ranker_weights.get(name, 0.0) > 0
        }
        if not active:
            raise ValueError("At least one enabled retriever must have a positive weight.")
        total = sum(active.values()) or 1.0
        return {name: weight / total for name, weight in active.items()}

    def export_ranker_settings(
        self,
        weights: Dict[str, float],
        enabled_retrievers: List[str] | None = None,
    ) -> Dict[str, Dict[str, float] | List[str]]:
        enabled = enabled_retrievers or [name for name, weight in weights.items() if float(weight) > 0.0]
        filtered = {name: float(weights.get(name, 0.0)) for name in sorted(SUPPORTED_RETRIEVERS)}
        total = sum(filtered.get(name, 0.0) for name in enabled) or 1.0
        normalized = {
            name: (filtered.get(name, 0.0) / total) if name in enabled else 0.0
            for name in sorted(SUPPORTED_RETRIEVERS)
        }
        return {
            "enabled_retrievers": list(enabled),
            "ranker_weights": normalized,
        }

    @staticmethod
    def _resolve_model_candidate(preferred: str, fallback: str) -> tuple[str, bool]:
        if preferred == fallback:
            return fallback, False
        if gguf_path_available(preferred):
            return preferred, False
        return fallback, True

    def resolve_runtime_model_profile(self) -> str:
        if self.runtime_model_profile == AUTO_MODEL_PROFILE:
            return GPU_MODEL_PROFILE if detect_hardware_profile().gpu_available else BASELINE_MODEL_PROFILE
        return self.runtime_model_profile

    def resolve_runtime_models(self, *, gen_model_override: str | None = None) -> Dict[str, object]:
        requested_profile = self.runtime_model_profile
        selected_profile = self.resolve_runtime_model_profile()
        hardware = detect_hardware_profile()

        if selected_profile == GPU_MODEL_PROFILE:
            embed_model, embed_fallback = self._resolve_model_candidate(
                self.gpu_embed_model,
                self.embed_model,
            )
            gen_model, gen_fallback = self._resolve_model_candidate(
                gen_model_override or self.gpu_gen_model,
                gen_model_override or self.gen_model,
            )
        else:
            embed_model, embed_fallback = self.embed_model, False
            gen_model, gen_fallback = gen_model_override or self.gen_model, False

        final_profile = (
            BASELINE_MODEL_PROFILE
            if selected_profile == GPU_MODEL_PROFILE and (embed_fallback or gen_fallback)
            else selected_profile
        )
        return {
            "requested_profile": requested_profile,
            "selected_profile": final_profile,
            "hardware_backend": hardware.backend,
            "hardware_reason": hardware.reason,
            "embed_model": embed_model,
            "gen_model": gen_model,
            "gpu_model_missing": embed_fallback or gen_fallback,
        }
    
    def get_config_state(self) -> None:
        """Returns dict of all config parameters except chunk_config """
        state = self.__dict__.copy()
        state.pop("chunk_config", None) # remove chunk_config to avoid serialization issues
        # also pop any non-serializable fields if needed
        for key in list(state.keys()):
            if not isinstance(state[key], (int, float, str, bool, list, dict, type(None))):
                state.pop(key)
        return state
        
