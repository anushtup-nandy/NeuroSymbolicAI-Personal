"""
Knowledge Graph Implementation
===============================
Multi-layer knowledge graph with semantic embeddings and analytics.
"""

import networkx as nx
import numpy as np
from typing import Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from modules.core.types import NodeType, RelationType, NodeID, Metadata


class PersonalGraph:
    """
    Multi-layer knowledge graph with semantic embeddings and analytics.
    
    Layers:
    1. Explicit: Obsidian [[links]], email threads
    2. Semantic: Cosine similarity > threshold
    3. Temporal: Time-ordered events
    
    Attributes:
        graph: NetworkX directed graph storing nodes and edges
        model: Sentence-BERT model for embeddings
        embeddings: Dict mapping node_id to embedding vector
        metadata: Dict mapping node_id to metadata dict
    """
    
    def __init__(self, model: SentenceTransformer):
        """
        Initialize the knowledge graph.
        
        Args:
            model: Pre-loaded Sentence-BERT model for embeddings
        """
        self.graph = nx.DiGraph()
        self.model = model
        self.embeddings = {}  # node_id -> embedding vector
        self.metadata = {}    # node_id -> {created, source, type, ...}
        
    def add_node(self, node_id: NodeID, node_type: NodeType, content: str, 
                 metadata: Optional[Metadata] = None):
        """
        Add node with automatic semantic embedding.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (from NodeType enum)
            content: Text content for embedding generation
            metadata: Optional metadata dict
        """
        clean_id = node_id.strip()
        
        if clean_id not in self.graph:
            # Compute embedding
            embedding = self.model.encode(content, convert_to_numpy=True)
            
            # Store
            self.graph.add_node(clean_id, type=node_type, content=content)
            self.embeddings[clean_id] = embedding
            self.metadata[clean_id] = metadata or {}
            
    def add_edge(self, source: NodeID, target: NodeID, relation: RelationType, 
                 weight: float = 1.0, metadata: Optional[Metadata] = None):
        """
        Add weighted edge with metadata.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation: Type of relationship
            weight: Edge weight (default 1.0)
            metadata: Optional edge metadata
        """
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, 
                              relation=relation, 
                              weight=weight,
                              metadata=metadata or {})
    
    def build_semantic_layer(self, threshold: float = 0.7) -> int:
        """
        Connect nodes with high semantic similarity.
        
        Creates edges between nodes whose embeddings have cosine similarity
        above the threshold.
        
        Args:
            threshold: Minimum similarity to create edge (0.0 to 1.0)
            
        Returns:
            Number of semantic edges added
        """
        nodes = list(self.graph.nodes())
        embeddings_matrix = np.array([self.embeddings[n] for n in nodes])
        
        # Compute all pairwise similarities
        similarities = cosine_similarity(embeddings_matrix)
        
        edges_added = 0
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i < j and similarities[i, j] > threshold:
                    self.add_edge(node_i, node_j, 
                                RelationType.RELATED_TO, 
                                weight=float(similarities[i, j]))
                    edges_added += 1
                    
        return edges_added
    
    def export_stats(self) -> Dict:
        """
        Export graph statistics.
        
        Returns:
            Dictionary with num_nodes, num_edges, density, avg_degree
        """
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        
        if num_nodes == 0:
            return {'num_nodes': 0, 'num_edges': 0, 'density': 0.0, 'avg_degree': 0.0}
        
        # Correct density for directed graph
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0.0
        
        # Average degree
        degrees = [d for _, d in self.graph.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': round(density, 6),
            'avg_degree': round(avg_degree, 2)
        }
