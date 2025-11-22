"""
Core Type Definitions for Decision Intelligence Engine
========================================================
Centralized enums and type definitions used throughout the system.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    CONCEPT = "concept"
    PERSON = "person"
    ORG = "organization"
    DATE = "date"
    LOCATION = "location"
    RESOURCE = "resource"
    RULE = "heuristic"
    VALUE = "value"
    DECISION = "decision"
    EVENT = "event"


class RelationType(str, Enum):
    """Types of relationships between nodes."""
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PRECEDES = "precedes"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    LINKS_TO = "links_to"  # Obsidian [[links]]
    MENTIONS = "mentions"


# Type aliases for clarity
NodeID = str
Embedding = Any  # numpy array
Metadata = Dict[str, Any]
