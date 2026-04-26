import numpy as np
import pytest

from src.embedder import SentenceTransformer


pytestmark = pytest.mark.unit


def test_encode_batch_multi_process_splits_work_across_pool(monkeypatch):
    observed_chunks = []

    def fake_encode_batch_worker(texts):
        observed_chunks.append(list(texts))
        return [[float(len(text))] for text in texts]

    class FakePool:
        _processes = 3

        def map(self, func, chunks):
            return [func(chunk) for chunk in chunks]

    monkeypatch.setattr("src.embedder._encode_batch_worker", fake_encode_batch_worker)

    embedder = object.__new__(SentenceTransformer)
    embedder._embedding_dimension = 1

    vectors = SentenceTransformer.encode_batch_multi_process(
        embedder,
        ["a", "bb", "ccc", "dddd", "eeeee"],
        FakePool(),
    )

    assert observed_chunks == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert np.array_equal(vectors, np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32))


def test_encode_batch_falls_back_to_single_text_requests(monkeypatch):
    class FakeModel:
        def create_embedding(self, texts):
            if isinstance(texts, list):
                raise RuntimeError("batch failure")
            return {"data": [{"embedding": [float(len(texts))]}]}

    embedder = object.__new__(SentenceTransformer)
    embedder.model = FakeModel()
    embedder._embedding_dimension = 1

    vectors = SentenceTransformer.encode_batch(embedder, ["a", "bb", "ccc"])

    assert np.array_equal(vectors, np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
