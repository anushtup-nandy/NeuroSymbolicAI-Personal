"""Decision Intelligence Engine - Modular Architecture."""

__version__ = "2.0.0"
__author__ = "Code Partner AI"

# Core exports
from modules.core import NodeType, RelationType
from modules.graph import PersonalGraph, analytics
from modules.ingestion import ObsidianIngestor, GoogleTakeoutIngestor
from modules.heuristics import DecisionModels
from modules.models import CausalInference

__all__ = [
    'NodeType',
    'RelationType',
    'PersonalGraph',
    'analytics',
    'ObsidianIngestor',
    'GoogleTakeoutIngestor',
    'DecisionModels',
    'CausalInference'
]
