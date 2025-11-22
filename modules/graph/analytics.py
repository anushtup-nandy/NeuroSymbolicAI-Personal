"""
Graph Analytics and Search
===========================
Advanced analytics functions for knowledge graph exploration.
"""

import networkx as nx
import numpy as np
import graphviz
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

from modules.core.types import NodeType, RelationType


def semantic_search(graph_obj, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
    """
    Semantic search using dense embeddings.
    
    Args:
        graph_obj: PersonalGraph instance
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of (node_id, similarity_score, node_data) tuples
    """
    if not graph_obj.embeddings:
        return []
    
    # Encode query
    query_embedding = graph_obj.model.encode(query, convert_to_numpy=True)
    
    # Compute similarities
    nodes = list(graph_obj.embeddings.keys())
    embeddings_matrix = np.array([graph_obj.embeddings[n] for n in nodes])
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
    
    # Get top-k
    top_indices = similarities.argsort()[:-top_k-1:-1]
    results = [(nodes[i], float(similarities[i]), graph_obj.graph.nodes[nodes[i]]) 
               for i in top_indices if similarities[i] > 0.1]
    
    return results


def find_path(graph_obj, source: str, target: str) -> Optional[List[str]]:
    """
    Find shortest path between two concepts.
    
    Args:
        graph_obj: PersonalGraph instance
        source: Source node ID
        target: Target node ID
        
    Returns:
        List of node IDs forming the path, or None if no path exists
    """
    try:
        return nx.shortest_path(graph_obj.graph.to_undirected(), source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def get_influential_nodes(graph_obj, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Get most influential nodes via PageRank.
    
    Args:
        graph_obj: PersonalGraph instance
        top_k: Number of top nodes to return
        
    Returns:
        List of (node_id, pagerank_score) tuples
    """
    if len(graph_obj.graph) == 0:
        return []
    
    pagerank = nx.pagerank(graph_obj.graph, weight='weight')
    sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_k]


def detect_communities(graph_obj) -> List[set]:
    """
    Detect thought clusters via Louvain community detection.
    
    Args:
        graph_obj: PersonalGraph instance
        
    Returns:
        List of sets, each containing node IDs in a community
    """
    if len(graph_obj.graph) < 3:
        return []
    
    undirected = graph_obj.graph.to_undirected()
    communities = nx.community.louvain_communities(undirected, weight='weight')
    return communities


def detect_contradictions(graph_obj, concept: str, window: int = 2) -> List[Dict]:
    """
    Detect contradictory statements about a concept.
    
    Strategy:
    1. Find all nodes mentioning the concept
    2. Check for opposing sentiment/stance
    3. Return pairs with high semantic similarity but opposing views
    
    Args:
        graph_obj: PersonalGraph instance
        concept: Concept to analyze
        window: Context window (unused, kept for API compatibility)
        
    Returns:
        List of contradiction dicts with node pairs and similarity
    """
    contradictions = []
    
    # Find related nodes
    related = semantic_search(graph_obj, concept, top_k=20)
    
    # Simple heuristic: look for negation words in similar contexts
    positive_words = {'good', 'great', 'positive', 'beneficial', 'prefer', 'love'}
    negative_words = {'bad', 'negative', 'avoid', 'harmful', 'dislike', 'hate', 'never'}
    
    for i, (node_i, sim_i, data_i) in enumerate(related):
        content_i = data_i['content'].lower()
        has_pos_i = any(w in content_i for w in positive_words)
        has_neg_i = any(w in content_i for w in negative_words)
        
        for j, (node_j, sim_j, data_j) in enumerate(related):
            if i >= j:
                continue
                
            content_j = data_j['content'].lower()
            has_pos_j = any(w in content_j for w in positive_words)
            has_neg_j = any(w in content_j for w in negative_words)
            
            # Contradiction: high similarity, opposite sentiment
            if sim_i > 0.6 and sim_j > 0.6:
                if (has_pos_i and has_neg_j) or (has_neg_i and has_pos_j):
                    contradictions.append({
                        'node_1': node_i,
                        'node_2': node_j,
                        'similarity': (sim_i + sim_j) / 2,
                        'content_1': content_i[:100],
                        'content_2': content_j[:100]
                    })
    
    return contradictions[:5]  # Top 5


def visualize_subgraph(graph_obj, central_nodes: List[str], depth: int = 2):
    """
    Create Graphviz visualization around central nodes.
    
    Args:
        graph_obj: PersonalGraph instance
        central_nodes: List of central node IDs to visualize around
        depth: How many hops from central nodes to include
        
    Returns:
        Graphviz Digraph object
    """
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', size='10,8')
    
    # Collect subgraph
    subgraph_nodes = set(central_nodes)
    for node in central_nodes:
        if node in graph_obj.graph:
            # Add neighbors up to depth
            for _ in range(depth):
                neighbors = (set(graph_obj.graph.successors(node)) | 
                           set(graph_obj.graph.predecessors(node)))
                subgraph_nodes.update(neighbors)
    
    # Add nodes
    for node in subgraph_nodes:
        if node not in graph_obj.graph:
            continue
            
        node_type = graph_obj.graph.nodes[node].get('type', NodeType.CONCEPT)
        
        # Style by type
        if node in central_nodes:
            shape, color = 'doubleoctagon', 'lightblue'
        elif node_type == NodeType.RULE:
            shape, color = 'box', 'lightyellow'
        elif node_type == NodeType.PERSON:
            shape, color = 'ellipse', 'lightgreen'
        else:
            shape, color = 'ellipse', 'lightgray'
        
        label = node[:30] + "..." if len(node) > 30 else node
        dot.node(node, label, shape=shape, style='filled', fillcolor=color)
    
    # Add edges
    for source, target in graph_obj.graph.edges():
        if source in subgraph_nodes and target in subgraph_nodes:
            relation = graph_obj.graph[source][target].get('relation', '')
            dot.edge(source, target, label=relation)
    
    return dot
