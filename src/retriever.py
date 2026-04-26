"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import pathlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import nltk
from nltk.stem import WordNetLemmatizer

import faiss
import numpy as np
from src.embedder import CachedEmbedder

from src.config import RAGConfig
from src.graph.retrieval import GraphRetriever
from src.graph.store import GraphStore, load_graph_store
from src.index_builder import preprocess_for_bm25


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, CachedEmbedder] = {}

def _get_embedder(model_name: str) -> CachedEmbedder:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = CachedEmbedder(model_name)
    return _EMBED_CACHE[model_name]


# -------------------------- Read artifacts -------------------------------

def load_artifacts(
    artifacts_dir: os.PathLike,
    index_prefix: str,
    cfg: Optional[RAGConfig] = None,
) -> Tuple[faiss.Index, Any, List[str], List[str], Any, Optional[GraphStore]]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    faiss_index = faiss.read_index(str(artifacts_dir / f"{index_prefix}.faiss"))
    bm25_index  = pickle.load(open(artifacts_dir / f"{index_prefix}_bm25.pkl", "rb"))
    chunks      = pickle.load(open(artifacts_dir / f"{index_prefix}_chunks.pkl", "rb"))
    sources     = pickle.load(open(artifacts_dir / f"{index_prefix}_sources.pkl", "rb"))
    metadata = pickle.load(open(artifacts_dir / f"{index_prefix}_meta.pkl", "rb"))
    graph_store = None

    graph_path = None
    require_graph = False
    if cfg is not None:
        require_graph = "graph" in cfg.get_active_ranker_weights()
        graph_path = cfg.get_graph_artifact_path(index_prefix, artifacts_dir=artifacts_dir)
    else:
        candidate_path = artifacts_dir / f"{index_prefix}_graph.json"
        if candidate_path.exists():
            graph_path = candidate_path

    if graph_path is not None and pathlib.Path(graph_path).exists():
        graph_store = load_graph_store(pathlib.Path(graph_path))
    elif require_graph:
        raise FileNotFoundError(f"Graph artifacts required but missing: {graph_path}")

    return faiss_index, bm25_index, chunks, sources, metadata, graph_store


# -------------------------- Helper to get page nums for chunks -------------------------------

def get_page_numbers(chunk_indices: list[int], metadata: list[dict]) -> dict[int, List[int]]:
    if not metadata or not chunk_indices:
        return {}

    page_map: dict[int, List[int]] = {}

    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)
        if 0 <= chunk_idx < len(metadata):
            chunk_pages = metadata[chunk_idx].get("page_numbers")
            if chunk_pages is None:
                continue  # don't store None; callers can default to [1]
            page_map[chunk_idx] = chunk_pages

    return page_map

# -------------------------- Filtering logic -----------------------------

def filter_retrieved_chunks(cfg: RAGConfig, chunks, ordered):
    topk_idxs = ordered[:cfg.top_k]
    return topk_idxs

# -------------------------- Retrieval core ------------------------------

class Retriever(ABC):
    @abstractmethod
    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        """Retrieves the top 'pool_size' chunks cores for a given query."""
        pass


def _resolve_page_to_chunk_map_path(
    cfg: RAGConfig,
    artifacts_dir: os.PathLike,
    index_prefix: str,
) -> pathlib.Path:
    configured_path = pathlib.Path(cfg.page_to_chunk_map_path)
    if configured_path.exists():
        return configured_path
    return pathlib.Path(artifacts_dir) / f"{index_prefix}_page_to_chunk_map.json"


def build_retrievers(
    cfg: RAGConfig,
    *,
    faiss_index: Any,
    bm25_index: Any,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    graph_store: Optional[GraphStore] = None,
) -> List["Retriever"]:
    retrievers: List[Retriever] = []
    enabled = cfg.get_active_ranker_weights()

    if "faiss" in enabled:
        retrievers.append(FAISSRetriever(faiss_index, cfg.embed_model))
    if "bm25" in enabled:
        retrievers.append(BM25Retriever(bm25_index))
    if "index_keywords" in enabled:
        retrievers.append(
            IndexKeywordRetriever(
                cfg.extracted_index_path,
                _resolve_page_to_chunk_map_path(cfg, artifacts_dir, index_prefix),
            )
        )
    if "graph" in enabled:
        if graph_store is None:
            raise ValueError("Graph retriever is enabled but graph artifacts were not loaded.")
        retrievers.append(GraphRetriever(graph_store))

    return retrievers


class FAISSRetriever(Retriever):
    name = "faiss"

    def __init__(self, index, embed_model: str):
        self.index = index
        self.embedder = _get_embedder(embed_model)

    def get_scores(self,
                query: str,
                pool_size: int,
                chunks: List[str]) -> Dict[int, float]:
        """
        Returns FAISS scores for top 'pool_size' keyed by global chunk index.
        """
        # FAISS expects a 2D array
        q_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check on vector dimensions
        if q_vec.shape[1] !=  self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={ self.index.d} vs query={q_vec.shape[1]}"
            )

        # Perform the search
        distances, indices =  self.index.search(q_vec, pool_size)

        # Remove invalid indices and ensure they are within bounds
        cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

        # Create the distance dictionary, ensuring we only include valid candidates
        dists = {idx: float(dist) for idx, dist in zip(cand_idxs, distances[0][:len(cand_idxs)])}

        # Invert distance to score: 1 / (1 + distance). Adding 1 avoids division by zero.
        return {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in dists.items()
        }


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index):
        self.index = index

    def get_scores(self,
                 query: str,
                 pool_size: int,
                 chunks: List[str]) -> Dict[int, float]:
        """
        Returns BM25 scores for top 'pool_size' keyed by global chunk index.
        """
        # Tokenize the query in the same way the index was built
        tokenized_query = preprocess_for_bm25(query)

        # Get scores for all documents in the corpus
        all_scores = self.index.get_scores(tokenized_query)

        # Find the indices of the top 'pool_size' scores
        num_candidates = min(pool_size, len(all_scores))
        top_k_indices = np.argpartition(-all_scores, kth=num_candidates-1)[:num_candidates]

        # Remove invalid indices and ensure they are within bounds
        top_k_indices = [i for i in top_k_indices if 0 <= i < len(chunks)]
        
        # Get the corresponding scores for the top indices
        top_scores = all_scores[top_k_indices]

        # Format the output as a dictionary of scores
        scores = {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}

        return scores


class IndexKeywordRetriever(Retriever):
    name = "index_keywords"
    
    def __init__(self, extracted_index_path: os.PathLike, page_to_chunk_map_path: os.PathLike):
        """
        Retriever that uses textbook index keywords to boost chunks on relevant pages.
        
        Args:
            extracted_index_path: Path to extracted_index.json (keyword -> page numbers)
            page_to_chunk_map_path: Path to page_to_chunk_map.json (page -> chunk IDs)
        """
        import json
        nltk.download('wordnet', quiet=True)
        self.page_to_chunk_map = {}
        
        # Load and normalize index: lemmatize phrases as units
        # Build token->phrase mapping for fast lookup
        if os.path.exists(extracted_index_path):
            lemmatizer = WordNetLemmatizer()
            
            with open(extracted_index_path, 'r') as f:
                raw_index = json.load(f)
                self.phrase_to_pages = {}  # phrase -> pages
                self.token_to_phrases = {}  # token -> [phrases]
                
                for key, pages in raw_index.items():
                    # Lemmatize each word in the phrase but keep phrase together
                    key_lower = key.lower()
                    words = key_lower.split()
                    lemmatized_words = []
                    
                    for word in words:
                        cleaned = word.strip('.,!?()[]:"\'')
                        if not cleaned:
                            continue
                        lemmatized_words.append(self._lemmatize_word(cleaned, lemmatizer))
                    
                    lemmatized_phrase = ' '.join(lemmatized_words)
                    self.phrase_to_pages[lemmatized_phrase] = pages
                    
                    # Build reverse index: each token points to phrases containing it
                    for token in lemmatized_words:
                        if token not in self.token_to_phrases:
                            self.token_to_phrases[token] = []
                        self.token_to_phrases[token].append(lemmatized_phrase)
        else:
            self.phrase_to_pages = {}
            self.token_to_phrases = {}
        
        if os.path.exists(page_to_chunk_map_path):
            with open(page_to_chunk_map_path, 'r') as f:
                self.page_to_chunk_map = json.load(f)
    
    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        """
        Returns scores for chunks that match index keywords.
        Score is proportional to the number of keyword hits.
        """
        keywords = self._extract_keywords(query)
        # chunk_id -> hit count
        chunk_hit_counts: Dict[int, int] = {} 
        
        # Match query keywords against index phrases (token overlap)
        for keyword in keywords:
            if keyword not in self.token_to_phrases:
                continue
            
            # Get all phrases containing this keyword token
            matching_phrases = self.token_to_phrases[keyword]
            
            for phrase in matching_phrases:
                page_numbers = self.phrase_to_pages[phrase]
                
                # Map pages to chunks
                for page_no in page_numbers:
                    chunk_ids = self.page_to_chunk_map.get(str(page_no), [])
                    for chunk_id in chunk_ids:
                        if chunk_id >= 0 and chunk_id < len(chunks):
                            chunk_hit_counts[chunk_id] = chunk_hit_counts.get(chunk_id, 0) + 1
        
        if not chunk_hit_counts:
            return {}
        
        # Normalize scores: more keyword hits = higher score
        max_hits = max(chunk_hit_counts.values())
        scores = {
            chunk_id: float(hit_count) / max_hits
            for chunk_id, hit_count in chunk_hit_counts.items()
        }
        
        return scores
    
    @staticmethod
    def _lemmatize_word(word: str, lemmatizer) -> str:
        """Lemmatize a word, trying noun then verb."""
        try:
            lemma = lemmatizer.lemmatize(word, pos='n')
            if lemma == word:
                lemma = lemmatizer.lemmatize(word, pos='v')
            return lemma
        except Exception:
            # Fallback when local WordNet data is unavailable or corrupted.
            return word
    
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract keywords from query by removing stopwords and lemmatizing."""
        
        stopwords = {
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", 
            "what", "how", "why", "when", "where", "who", "does", "do", "be"
        }
        
        lemmatizer = WordNetLemmatizer()
        words = query.lower().split()
        keywords = []
        for word in words:
            cleaned = word.strip('.,!?()[]:"\'')
            if not cleaned or cleaned in stopwords:
                continue
            keywords.append(IndexKeywordRetriever._lemmatize_word(cleaned, lemmatizer))
        return keywords
