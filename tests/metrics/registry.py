from typing import Dict, List, Optional
from tests.metrics.base import MetricBase


class MetricRegistry:
    """Registry for managing available metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, MetricBase] = {}
        self._auto_register()
    
    def _auto_register(self):
        """Automatically register all available metrics."""
        from tests.metrics import (
            SemanticSimilarityMetric,
            KeywordMatchMetric,
            NLIEntailmentMetric,
            AsyncLLMJudgeMetric,
            ChunkRetrievalMetric
        )

        self._safe_register(SemanticSimilarityMetric)
        self._safe_register(KeywordMatchMetric)
        self._safe_register(NLIEntailmentMetric)
        self._safe_register(AsyncLLMJudgeMetric)
        self._safe_register(ChunkRetrievalMetric)

    def _safe_register(self, metric_cls):
        """Skip optional metrics that cannot initialize in the current environment."""
        try:
            self.register(metric_cls())
        except Exception as exc:
            print(f"Skipping metric {metric_cls.__name__}: {exc}")

    def register(self, metric: MetricBase):
        """Register a new metric."""
        self._metrics[metric.name] = metric
        print(f"Registered metric: {metric}")
    
    def get_metric(self, name: str) -> Optional[MetricBase]:
        """Get a metric by name."""
        return self._metrics.get(name)
    
    def get_available_metrics(self) -> Dict[str, MetricBase]:
        """Get all available metrics that can be used."""
        return {name: metric for name, metric in self._metrics.items() 
                if metric.is_available()}
    
    def get_all_metrics(self) -> Dict[str, MetricBase]:
        """Get all registered metrics (including unavailable ones)."""
        return self._metrics.copy()
    
    def list_metric_names(self) -> List[str]:
        """List all available metric names."""
        return list(self.get_available_metrics().keys())
    
    def list_all_metric_names(self) -> List[str]:
        """List all registered metric names (including unavailable)."""
        return list(self._metrics.keys())
