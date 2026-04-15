from unittest.mock import Mock, patch

import pytest

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import build_retrievers


@pytest.mark.unit
def test_build_retrievers_loads_enabled_three_source_hybrid(tmp_path):
    index_path = tmp_path / "index.json"
    index_path.write_text('{"database": [1]}')
    page_map_path = tmp_path / "page_map.json"
    page_map_path.write_text('{"1": [0]}')

    cfg = RAGConfig(
        ranker_weights={
            "faiss": 0.55,
            "bm25": 0.3,
            "index_keywords": 0.15,
        },
        enabled_retrievers={
            "faiss": True,
            "bm25": True,
            "index_keywords": True,
        },
        extracted_index_path=str(index_path),
        page_to_chunk_map_path=str(page_map_path),
    )

    with patch("src.retriever._get_embedder"), patch("src.retriever.WordNetLemmatizer") as mock_lemmatizer_cls:
        mock_lemmatizer = Mock()
        mock_lemmatizer.lemmatize.side_effect = lambda word, pos=None: word
        mock_lemmatizer_cls.return_value = mock_lemmatizer

        retrievers = build_retrievers(
            cfg,
            faiss_index=object(),
            bm25_index=object(),
        )

    assert [retriever.name for retriever in retrievers] == [
        "faiss",
        "bm25",
        "index_keywords",
    ]


@pytest.mark.unit
def test_rrf_fusion_is_deterministic_with_ties():
    ranker = EnsembleRanker(
        ensemble_method="rrf",
        weights={"faiss": 0.7, "bm25": 0.3},
        active_retrievers=["faiss", "bm25"],
    )

    ordered_ids, ordered_scores = ranker.rank(
        {
            "faiss": {1: 1.0, 0: 1.0},
            "bm25": {0: 1.0, 1: 1.0},
        }
    )

    assert ordered_ids == [0, 1]
    assert ordered_scores[0] >= ordered_scores[1]


@pytest.mark.unit
def test_linear_fusion_rewards_multi_source_agreement():
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 0.55, "bm25": 0.3, "index_keywords": 0.15},
        active_retrievers=["faiss", "bm25", "index_keywords"],
    )

    ordered_ids, ordered_scores = ranker.rank(
        {
            "faiss": {0: 0.95, 1: 0.8},
            "bm25": {1: 2.0, 2: 1.0},
            "index_keywords": {1: 1.0},
        }
    )

    assert ordered_ids[0] == 1
    assert ordered_scores[0] > ordered_scores[1]


@pytest.mark.unit
def test_linear_constant_scores_keep_sources_active():
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 0.55, "bm25": 0.3, "index_keywords": 0.15},
        active_retrievers=["faiss", "bm25", "index_keywords"],
    )

    ordered_ids, _ = ranker.rank(
        {
            "faiss": {0: 1.0, 1: 1.0},
            "bm25": {1: 3.0},
            "index_keywords": {1: 1.0},
        }
    )

    assert ordered_ids[0] == 1
