"""
Base Ingestion Pipeline
========================
Shared utilities for entity and relation extraction.
"""

import re
import streamlit as st
from datetime import datetime
from typing import List, Tuple
from collections import defaultdict

from modules.core.types import NodeType, RelationType


class BaseIngestor:
    """
    Base class for data ingestion with shared extraction logic.
    
    Provides:
    - Entity extraction (spaCy NER + regex)
    - Relation extraction (causal patterns, co-occurrence)
    - Logging infrastructure
    - Stats tracking
    """
    
    def __init__(self, graph, spacy_model, ollama_model: str = "gemma3:4b"):
        """
        Initialize the base ingestor.
        
        Args:
            graph: PersonalGraph instance to populate
            spacy_model: Pre-loaded spaCy model
            ollama_model: Ollama model name for LLM extraction (optional)
        """
        self.graph = graph
        self.nlp = spacy_model
        self.ollama_model = ollama_model
        self.stats = defaultdict(int)
        
    def log(self, msg: str):
        """
        Log message to session state.
        
        Args:
            msg: Message to log
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.session_state['ingestion_log'].append(f"{timestamp} | {msg}")
        
    def extract_entities(self, text: str) -> List[Tuple[str, NodeType]]:
        """
        Extract entities using spaCy + regex fallbacks.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of (entity_text, entity_type) tuples
        """
        entities = []
        
        # spaCy NER
        doc = self.nlp(text[:100000])  # Limit for performance
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append((ent.text, NodeType.PERSON))
            elif ent.label_ == "ORG":
                entities.append((ent.text, NodeType.ORG))
            elif ent.label_ == "DATE":
                entities.append((ent.text, NodeType.DATE))
            elif ent.label_ == "GPE":
                entities.append((ent.text, NodeType.LOCATION))
        
        # Regex for common patterns
        # Money
        money_pattern = r'\$\d+[kKmMbB]?'
        for match in re.finditer(money_pattern, text):
            entities.append((match.group(), NodeType.CONCEPT))
        
        # URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append((match.group(), NodeType.RESOURCE))
        
        return entities
    
    def extract_relations(self, text: str, entities: List[Tuple[str, NodeType]]) -> List[Tuple[str, RelationType, str]]:
        """
        Extract relations between entities.
        
        Uses:
        - Causal keywords (because, led to, caused)
        - Temporal keywords (before, after)
        - Co-occurrence within window
        
        Args:
            text: Text to extract relations from
            entities: List of entities found in the text
            
        Returns:
            List of (source, relation_type, target) tuples
        """
        relations = []
        
        # Causal keywords
        causal_patterns = [
            (r'(.+?)\s+(?:because|since|due to)\s+(.+?)[.\n]', RelationType.CAUSES),
            (r'(.+?)\s+(?:led to|resulted in|caused)\s+(.+?)[.\n]', RelationType.CAUSES),
            (r'(.+?)\s+(?:before|prior to)\s+(.+?)[.\n]', RelationType.PRECEDES),
        ]
        
        for pattern, rel_type in causal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source, target = match.groups()
                relations.append((source.strip()[:50], rel_type, target.strip()[:50]))
        
        # Co-occurrence within window
        entity_texts = [e[0] for e in entities]
        words = text.split()
        window_size = 20
        
        for i, word in enumerate(words):
            for ent1 in entity_texts:
                if ent1.lower() in word.lower():
                    # Look ahead
                    window = ' '.join(words[i:i+window_size])
                    for ent2 in entity_texts:
                        if ent1 != ent2 and ent2.lower() in window.lower():
                            relations.append((ent1, RelationType.RELATED_TO, ent2))
        
        return relations
