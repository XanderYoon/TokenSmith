import pytest
from unittest.mock import MagicMock, patch
import argparse
from typing import List, Dict, Any

from src.graph import GraphArtifact, GraphEdge, GraphNode
from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.instrumentation.logging import RunLogger

class MockRetriever:
    def __init__(self, name: str, scores: Dict[int, float]):
        self.name = name
        self.scores = scores
    
    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        # Return pre-defined scores regardless of query
        # Simulate only returning scores for valid chunk indices
        return self.scores


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*SwigPyPacked.*")
@pytest.mark.filterwarnings("ignore:.*SwigPyObject.*")
@pytest.mark.filterwarnings("ignore:.*swigvarlink.*")
def test_end_to_end_pipeline_stubbed():
    """
    Test the full RAG pipeline with stubbed LLM and Vector DB.
    This ensures that the pipeline logic works efficiently without
    requiring model files or incurring inference costs.
    """
    from src.main import get_answer
    
    cfg = RAGConfig(
        top_k=2,
        num_candidates=5,
        ensemble_method="linear",
        ranker_weights={"faiss": 0.5, "bm25": 0.5},
        chunk_mode="recursive_sections",
        use_hyde=False,
        disable_chunks=False,
        rerank_mode="none"
    )
    
    args = argparse.Namespace(
        system_prompt_mode="baseline",
        index_prefix="test_index"
    )
    
    # Setup Dummy Data
    chunks = [
        "Chunk 0: Python is a programming language.",
        "Chunk 1: The sky is blue.",
        "Chunk 2: Machine learning uses statistics.",
        "Chunk 3: Databases store data.",
        "Chunk 4: API stands for Application Programming Interface."
    ]
    sources = ["doc1", "doc1", "doc2", "doc3", "doc4"]
    
    # Setup Mock Retrievers
    faiss_scores = {0: 0.9, 2: 0.8, 1: 0.1, 3: 0.05, 4: 0.05}
    bm25_scores = {0: 0.8, 2: 0.9, 3: 0.2, 1: 0.05, 4: 0.05}
    
    retrievers = [
        MockRetriever("faiss", faiss_scores),
        MockRetriever("bm25", bm25_scores)
    ]
    
    # Setup Ranker (using real EnsembleRanker)
    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 0.5, "bm25": 0.5}
    )
    
    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": [{"page_numbers": [1]} for _ in chunks]
    }
    
    # Mock the Generator
    def mock_stream_generator():
        yield "This is a dummy response "
        yield "from the stubbed LLM."

    # Using patch as a context manager to mock 'src.main.answer'
    # side_effect ensures each call gets a fresh generator
    with patch("src.main.answer", side_effect=lambda *a, **k: mock_stream_generator()) as mock_answer_func:
        
        logger = RunLogger()
        console = MagicMock()
        question = "What is Python?"
        
        # First, run in test mode to inspect retrieval internals.
        ans, chunks_info, hyde_query = get_answer(
            question=question,
            cfg=cfg,
            args=args,
            logger=logger,
            console=console,
            artifacts=artifacts,
            is_test_mode=True,
        )

        # Then, run with test mode disabled to exercise the full pipeline (including logging).
        ans_prod = get_answer(
            question=question,
            cfg=cfg,
            args=args,
            logger=logger,
            console=console,
            artifacts=artifacts,
            is_test_mode=False,
        )

        
        # 7. Assertions
        
        # Check LLM output reconstruction
        expected_ans = "This is a dummy response from the stubbed LLM."
        assert ans == expected_ans
        assert ans_prod == expected_ans
        
        # Check retrieval happened correctly
        # We expect top_k=2.
        # Based on MockRanker logic:
        # Chunk 0: 0.9 + 0.8 = 1.7
        # Chunk 2: 0.8 + 0.9 = 1.7
        # Chunk 1: 0.1 + 0.05 = 0.15
        # Chunk 3: 0.05 + 0.2 = 0.25
        # Top 2 should be 0 and 2 (order might depend on sort stability if equal, but they are top 2)
        
        assert len(chunks_info) == 2
        retrieved_chunk_ids = {info["chunk_id"] for info in chunks_info}
        assert 0 in retrieved_chunk_ids
        assert 2 in retrieved_chunk_ids
        
        # Check chunk content is correct in info
        for info in chunks_info:
            cid = info["chunk_id"]
            assert info["content"] == chunks[cid]
        
        # Check that answer() was called with correct context
        assert mock_answer_func.call_count == 2
        
        # Inspect arguments passed to answer()
        # answer(query, chunks, model_path, max_tokens, system_prompt_mode, temperature)
        call_args = mock_answer_func.call_args
        passed_query = call_args[0][0]     # positional arg 1: query
        passed_chunks = call_args[0][1]    # positional arg 2: chunks LIST
        
        assert passed_query == question
        assert len(passed_chunks) == 2
        # Check that the passed chunks contain the text we expect
        # Note: chunks might be passed exactly as they are in the 'chunks' list
        assert any("Python is a programming language." in c for c in passed_chunks)
        assert any("Machine learning uses statistics." in c for c in passed_chunks)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*SwigPyPacked.*")
@pytest.mark.filterwarnings("ignore:.*SwigPyObject.*")
@pytest.mark.filterwarnings("ignore:.*swigvarlink.*")
def test_end_to_end_pipeline_with_graph_signal():
    from src.main import get_answer

    cfg = RAGConfig(
        top_k=2,
        num_candidates=5,
        ensemble_method="linear",
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
        chunk_mode="recursive_sections",
        use_hyde=False,
        disable_chunks=False,
        rerank_mode="none",
    )

    args = argparse.Namespace(
        system_prompt_mode="baseline",
        index_prefix="test_index",
    )

    chunks = [
        "Chunk 0: Database schema overview.",
        "Chunk 1: Foreign key references primary key.",
        "Chunk 2: Primary key uniquely identifies tuples.",
        "Chunk 3: Unrelated networking content.",
    ]
    sources = ["doc1", "doc1", "doc1", "doc2"]

    retrievers = [
        MockRetriever("faiss", {0: 0.9, 1: 0.5, 2: 0.2}),
        MockRetriever("bm25", {1: 1.0, 2: 0.8}),
        MockRetriever("index_keywords", {1: 1.0}),
        MockRetriever("graph", {1: 1.0, 2: 0.7}),
    ]

    ranker = EnsembleRanker(
        ensemble_method="linear",
        weights={"faiss": 0.45, "bm25": 0.25, "index_keywords": 0.15, "graph": 0.15},
        active_retrievers=["faiss", "bm25", "index_keywords", "graph"],
    )

    artifacts = {
        "chunks": chunks,
        "sources": sources,
        "retrievers": retrievers,
        "ranker": ranker,
        "meta": [{"page_numbers": [1]} for _ in chunks],
        "graph": GraphArtifact(
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
        ),
    }

    def mock_stream_generator():
        yield "Foreign keys reference "
        yield "primary keys."

    with patch("src.main.answer", side_effect=lambda *a, **k: mock_stream_generator()) as mock_answer_func:
        logger = RunLogger()
        console = MagicMock()
        question = "How does a foreign key reference a primary key?"

        ans, chunks_info, _ = get_answer(
            question=question,
            cfg=cfg,
            args=args,
            logger=logger,
            console=console,
            artifacts=artifacts,
            is_test_mode=True,
        )

        assert ans == "Foreign keys reference primary keys."
        assert len(chunks_info) == 2
        assert chunks_info[0]["chunk_id"] == 1
        assert chunks_info[0]["graph_score"] > 0.0
        assert all("graph_score" in info for info in chunks_info)
        assert any(info["graph_score"] > 0.0 for info in chunks_info)
        assert mock_answer_func.call_count == 1
