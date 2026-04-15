from unittest.mock import Mock, patch

import pytest

from src.graph import GraphArtifact, GraphEdge, GraphNode
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
def test_config_supports_four_source_hybrid_with_graph_disabled_until_wired():
    cfg = RAGConfig(
        ranker_weights={
            "faiss": 0.5,
            "bm25": 0.25,
            "index_keywords": 0.15,
            "graph": 0.1,
        },
        enabled_retrievers={
            "faiss": True,
            "bm25": True,
            "index_keywords": True,
            "graph": True,
        },
        graph_artifact_path="index/structure_aware/textbook_index_graph.json",
    )

    assert cfg.get_enabled_retriever_names() == [
        "faiss",
        "bm25",
        "index_keywords",
        "graph",
    ]
    assert cfg.get_active_ranker_weights()["graph"] > 0.0
    assert cfg.graph_artifact_path == "index/structure_aware/textbook_index_graph.json"


@pytest.mark.unit
def test_build_retrievers_loads_graph_retriever_when_enabled():
    cfg = RAGConfig(
        ranker_weights={
            "faiss": 0.45,
            "bm25": 0.25,
            "index_keywords": 0.15,
            "graph": 0.15,
        },
        enabled_retrievers={
            "faiss": True,
            "bm25": True,
            "index_keywords": True,
            "graph": True,
        },
    )

    graph_artifact = GraphArtifact(
        document_id="test_index",
        nodes=[
            GraphNode(node_id="entity:foreign-key", label="foreign key", chunk_ids=[1]),
            GraphNode(node_id="entity:primary-key", label="primary key", chunk_ids=[2]),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:foreign-key-references-primary-key",
                source_id="entity:foreign-key",
                target_id="entity:primary-key",
                relation="references",
                chunk_ids=[1],
            )
        ],
    )

    with patch("src.retriever._get_embedder"), patch("src.retriever.WordNetLemmatizer") as mock_lemmatizer_cls:
        mock_lemmatizer = Mock()
        mock_lemmatizer.lemmatize.side_effect = lambda word, pos=None: word
        mock_lemmatizer_cls.return_value = mock_lemmatizer

        retrievers = build_retrievers(
            cfg,
            faiss_index=object(),
            bm25_index=object(),
            graph_artifact=graph_artifact,
        )

    assert [retriever.name for retriever in retrievers] == [
        "faiss",
        "bm25",
        "index_keywords",
        "graph",
    ]


@pytest.mark.unit
def test_graph_retriever_scores_direct_and_neighbor_chunks():
    cfg = RAGConfig(
        ranker_weights={"graph": 1.0},
        enabled_retrievers={"graph": True},
    )
    graph_artifact = GraphArtifact(
        document_id="test_index",
        nodes=[
            GraphNode(node_id="entity:foreign-key", label="foreign key", chunk_ids=[1]),
            GraphNode(node_id="entity:primary-key", label="primary key", chunk_ids=[2]),
            GraphNode(node_id="entity:database-schema", label="database schema", chunk_ids=[0]),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:foreign-key-references-primary-key",
                source_id="entity:foreign-key",
                target_id="entity:primary-key",
                relation="references",
                chunk_ids=[1],
            )
        ],
    )

    retriever = build_retrievers(
        cfg,
        graph_artifact=graph_artifact,
    )[0]
    chunks = [
        "database schema overview",
        "foreign key references a primary key",
        "primary key uniquely identifies tuples",
    ]

    scores = retriever.get_scores("How does a foreign key reference a primary key?", 5, chunks)

    assert retriever.name == "graph"
    assert list(scores.keys())[0] == 1
    assert 2 in scores
    assert 0 not in scores


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


@pytest.mark.unit
def test_linear_fusion_accepts_four_source_hybrid():
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 0.45, "bm25": 0.25, "index_keywords": 0.15, "graph": 0.15},
        active_retrievers=["faiss", "bm25", "index_keywords", "graph"],
    )

    ordered_ids, ordered_scores = ranker.rank(
        {
            "faiss": {0: 0.9, 1: 0.8},
            "bm25": {1: 2.0, 2: 1.0},
            "index_keywords": {1: 1.0},
            "graph": {2: 1.0, 1: 0.6},
        }
    )

    assert ordered_ids[0] in {1, 2}
    assert ordered_scores[0] >= ordered_scores[1]
