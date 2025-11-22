"""
Triplet-Based Knowledge Extraction
===================================
Extract (Subject, Predicate, Object) triplets for semantic reasoning and Graph RAG.

Example triplets:
- (India, offers, family_proximity)
- (USA, requires, visa_approval)
- (visa_approval, status, delayed)

Enables causal chain reasoning:
USA → requires → visa_approval → status → delayed
"""

import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from modules.ingestion.llm_extractor import Triplet, LLMExtractor
from modules.core.types import RelationType


class TripletGraph:
    """
    Graph structure specifically for triplet-based reasoning.
    
    Optimized for:
    - Causal chain discovery
    - Multi-hop reasoning
    - Query answering via graph traversal
    """
    
    def __init__(self):
        """Initialize triplet graph."""
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        self.triplets: List[Triplet] = []
        self.predicate_index = defaultdict(list)  # predicate -> [(subject, object)]
        
    def add_triplet(self, triplet: Triplet):
        """
        Add a triplet to the graph.
        
        Args:
            triplet: Triplet object (subject, predicate, object)
        """
        self.triplets.append(triplet)
        
        # Add to graph
        self.graph.add_edge(
            triplet.subject,
            triplet.object,
            predicate=triplet.predicate,
            confidence=triplet.confidence
        )
        
        # Index by predicate for fast lookup
        self.predicate_index[triplet.predicate].append((triplet.subject, triplet.object))
    
    def find_causal_chains(self, 
                          start_entity: str, 
                          max_depth: int = 5,
                          min_confidence: float = 0.5) -> List[List[Tuple[str, str, str]]]:
        """
        Find causal chains starting from an entity.
        
        A causal chain is a path where each edge represents causation.
        
        Args:
            start_entity: Starting entity
            max_depth: Maximum chain length
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of chains, where each chain is [(subject, predicate, object), ...]
        
        Example:
            USA → requires → visa → delays → timeline_risk
            Returns: [
                [("USA", "requires", "visa"), 
                 ("visa", "delays", "timeline_risk")]
            ]
        """
        causal_predicates = {
            'causes', 'leads_to', 'results_in', 'triggers',
            'requires', 'needs', 'depends_on',
            'delays', 'blocks', 'prevents'
        }
        
        chains = []
        
        def dfs(current: str, path: List[Tuple], depth: int, visited: Set[str]):
            """Depth-first search for causal chains."""
            if depth > max_depth:
                return
            
            if path:  # Don't add empty paths
                chains.append(path.copy())
            
            # Get all outgoing edges
            if current not in self.graph:
                return
            
            for neighbor in self.graph.successors(current):
                if neighbor in visited:
                    continue  # Avoid cycles
                
                # Check all edges between current and neighbor
                for edge_data in self.graph[current][neighbor].values():
                    predicate = edge_data.get('predicate', '')
                    confidence = edge_data.get('confidence', 0.0)
                    
                    # Only follow causal predicates with sufficient confidence
                    if predicate in causal_predicates and confidence >= min_confidence:
                        new_visited = visited | {neighbor}
                        new_path = path + [(current, predicate, neighbor)]
                        dfs(neighbor, new_path, depth + 1, new_visited)
        
        dfs(start_entity, [], 0, {start_entity})
        
        # Sort by length (longer chains first)
        chains.sort(key=len, reverse=True)
        
        return chains
    
    def find_paths_between(self, 
                           source: str, 
                           target: str,
                           max_length: int = 4) -> List[List[str]]:
        """
        Find all paths between two entities.
        
        Args:
            source: Source entity
            target: Target entity  
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is list of entities)
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                source, 
                target, 
                cutoff=max_length
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_related_entities(self, 
                            entity: str, 
                            predicate: Optional[str] = None,
                            direction: str = 'both') -> List[Tuple[str, str, float]]:
        """
        Get entities related to a given entity.
        
        Args:
            entity: Entity to find relations for
            predicate: Filter by specific predicate (optional)
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of (related_entity, predicate, confidence) tuples
        """
        related = []
        
        if entity not in self.graph:
            return related
        
        # Outgoing edges (entity is subject)
        if direction in ('outgoing', 'both'):
            for neighbor in self.graph.successors(entity):
                for edge_data in self.graph[entity][neighbor].values():
                    pred = edge_data.get('predicate', '')
                    conf = edge_data.get('confidence', 0.0)
                    
                    if predicate is None or pred == predicate:
                        related.append((neighbor, pred, conf))
        
        # Incoming edges (entity is object)
        if direction in ('incoming', 'both'):
            for neighbor in self.graph.predecessors(entity):
                for edge_data in self.graph[neighbor][entity].values():
                    pred = edge_data.get('predicate', '')
                    conf = edge_data.get('confidence', 0.0)
                    
                    if predicate is None or pred == predicate:
                        related.append((neighbor, pred, conf))
        
        return related
    
    def answer_question(self, question: str) -> Dict[str, any]:
        """
        Answer questions using graph traversal.
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with answer, supporting triplets, and confidence
        
        Example:
            Q: "What does USA require?"
            A: {
                'answer': ['visa_approval', 'timeline_commitment'],
                'triplets': [('USA', 'requires', 'visa_approval')],
                'confidence': 0.9
            }
        """
        # Simple pattern matching (can be enhanced with LLM)
        question_lower = question.lower()
        
        # Extract entity from question (simple heuristic)
        entities_in_graph = set(self.graph.nodes())
        mentioned_entities = [e for e in entities_in_graph if e.lower() in question_lower]
        
        if not mentioned_entities:
            return {'answer': None, 'triplets': [], 'confidence': 0.0}
        
        entity = mentioned_entities[0]
        
        # Detect question type
        if 'require' in question_lower or 'need' in question_lower:
            predicate = 'requires'
        elif 'cause' in question_lower or 'lead' in question_lower:
            predicate = 'causes'
        elif 'conflict' in question_lower or 'oppose' in question_lower:
            predicate = 'conflicts_with'
        else:
            predicate = None
        
        # Get related entities
        related = self.get_related_entities(entity, predicate, direction='outgoing')
        
        answer_entities = [r[0] for r in related]
        supporting_triplets = [(entity, r[1], r[0]) for r in related]
        avg_confidence = sum(r[2] for r in related) / len(related) if related else 0.0
        
        return {
            'answer': answer_entities,
            'triplets': supporting_triplets,
            'confidence': avg_confidence
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get graph statistics."""
        return {
            'num_triplets': len(self.triplets),
            'num_entities': len(self.graph.nodes()),
            'num_unique_predicates': len(self.predicate_index),
            'avg_connections_per_entity': (
                len(self.triplets) / len(self.graph.nodes()) 
                if len(self.graph.nodes()) > 0 else 0
            )
        }


class TripletExtractor:
    """
    Extract triplets from text and build TripletGraph.
    """
    
    def __init__(self, llm_extractor: LLMExtractor):
        """
        Initialize triplet extractor.
        
        Args:
            llm_extractor: LLM extractor instance
        """
        self.llm_extractor = llm_extractor
        self.triplet_graph = TripletGraph()
        
    def extract_from_text(self, text: str, max_triplets: int = 20) -> List[Triplet]:
        """
        Extract triplets from text using LLM.
        
        Args:
            text: Text to extract from
            max_triplets: Maximum number of triplets
            
        Returns:
            List of Triplet objects
        """
        triplets = self.llm_extractor.extract_triplets(text, max_triplets)
        
        # Add to graph
        for triplet in triplets:
            self.triplet_graph.add_triplet(triplet)
        
        return triplets
    
    def get_graph(self) -> TripletGraph:
        """Get the accumulated triplet graph."""
        return self.triplet_graph
    
    def export_for_rag(self) -> List[Dict]:
        """
        Export triplets in format suitable for RAG.
        
        Returns:
            List of dicts with text representation of each triplet
        """
        return [
            {
                'text': f"{t.subject} {t.predicate} {t.object}",
                'subject': t.subject,
                'predicate': t.predicate,
                'object': t.object,
                'confidence': t.confidence
            }
            for t in self.triplet_graph.triplets
        ]
