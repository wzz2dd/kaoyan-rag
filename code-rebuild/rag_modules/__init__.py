from .data_preparation import DataLoaderModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule
from .generation_integration import GenerationIntegrationModule

# Backward compatibility alias
DataPreparationModule = DataLoaderModule

__all__ = [
    'DataLoaderModule',
    'DataPreparationModule',
    'IndexConstructionModule',
    'RetrievalOptimizationModule',
    'GenerationIntegrationModule'
]

__version__ = "2.0.0"