from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml
import pathlib

from src.preprocessing.chunking import (
    ChunkConfig,
    ChunkStrategy,
    SectionRecursiveConfig,
    SectionRecursiveStrategy,
    StructureAwareConfig,
    StructureAwareStrategy,
)

@dataclass
class RAGConfig:
    # chunking
    chunk_config: ChunkConfig = field(init=False)
    chunk_mode: str = "recursive_sections"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    oversize_fallback_overlap: int = 200

    # retrieval + ranking
    top_k: int = 10
    num_candidates: int = 60
    embed_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
    ensemble_method: str = "rrf"
    rrf_k: int  = 60
    ranker_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "faiss": 0.55,
            "bm25": 0.3,
            "index_keywords": 0.15,
        }
    )
    enabled_retrievers: Optional[Dict[str, bool]] = None
    score_normalization: str = "minmax"
    rerank_mode: str = ""
    rerank_top_k: int = 5

    # generation
    max_gen_tokens: int = 400
    gen_model: str = "models/qwen2.5-3b-instruct-q8_0.gguf"
    
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
        self.ranker_weights = {str(k): float(v) for k, v in self.ranker_weights.items()}
        self.enabled_retrievers = self._resolve_enabled_retrievers(
            enabled_retrievers=self.enabled_retrievers,
            weights=self.ranker_weights,
        )
        active_weights = self.get_active_ranker_weights()
        assert active_weights, "At least one retriever must be enabled with a positive weight"
        if self.ensemble_method.lower() in {"linear", "weighted", "rrf"}:
            total_weight = sum(active_weights.values())
            assert total_weight > 0, "Active retriever weights must sum to a positive value"
            self.ranker_weights = {
                name: (weight / total_weight if name in active_weights else float(weight))
                for name, weight in self.ranker_weights.items()
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
        if self.chunk_mode == "structure_aware":
            return StructureAwareConfig(
                max_chunk_chars=self.chunk_size,
                oversize_fallback_overlap=self.oversize_fallback_overlap,
            )
        raise ValueError(
            f"Unknown chunk_mode: {self.chunk_mode}. Supported: recursive_sections, structure_aware"
        )

    def get_chunk_strategy(self) -> ChunkStrategy:
        if isinstance(self.chunk_config, SectionRecursiveConfig):
            return SectionRecursiveStrategy(self.chunk_config)
        if isinstance(self.chunk_config, StructureAwareConfig):
            return StructureAwareStrategy(self.chunk_config)
        raise ValueError(f"Unknown chunk config type: {self.chunk_config.__class__.__name__}")

    def get_artifacts_directory(self) -> os.PathLike:
        """Returns the path prefix for index artifacts."""
        strategy = self.get_chunk_strategy()
        strategy_dir = pathlib.Path("index", strategy.artifact_folder_name())
        strategy_dir.mkdir(parents=True, exist_ok=True)
        return strategy_dir

    def get_enabled_retriever_names(self) -> List[str]:
        return [name for name, enabled in self.enabled_retrievers.items() if enabled]

    def get_active_ranker_weights(self) -> Dict[str, float]:
        return {
            name: float(self.ranker_weights.get(name, 0.0))
            for name in self.get_enabled_retriever_names()
            if float(self.ranker_weights.get(name, 0.0)) > 0
        }

    @staticmethod
    def _resolve_enabled_retrievers(
        enabled_retrievers: Optional[Dict[str, bool]],
        weights: Dict[str, float],
    ) -> Dict[str, bool]:
        supported = ("faiss", "bm25", "index_keywords")
        resolved = {name: False for name in supported}

        if enabled_retrievers is not None:
            for name, enabled in enabled_retrievers.items():
                if name not in resolved:
                    raise ValueError(f"Unknown retriever '{name}'. Supported: {supported}")
                resolved[name] = bool(enabled)
            return resolved

        for name in supported:
            resolved[name] = float(weights.get(name, 0.0)) > 0.0

        if not any(resolved.values()) and "faiss" in weights:
            resolved["faiss"] = True

        return resolved
    
    def get_config_state(self) -> None:
        """Returns dict of all config parameters except chunk_config """
        state = self.__dict__.copy()
        state.pop("chunk_config", None) # remove chunk_config to avoid serialization issues
        # also pop any non-serializable fields if needed
        for key in list(state.keys()):
            if not isinstance(state[key], (int, float, str, bool, list, dict, type(None))):
                state.pop(key)
        return state
        
