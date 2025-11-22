"""
Obsidian Vault Ingestion
=========================
Ingest Obsidian markdown files with entity and relation extraction.

v3.0 Enhancement: LLM-powered deep understanding with Gemma 3.
"""

import os
import glob
import re
import streamlit as st
from collections import defaultdict
from typing import Optional

from modules.core.types import NodeType, RelationType
from modules.ingestion.base import BaseIngestor


class ObsidianIngestor(BaseIngestor):
    """
    Ingest Obsidian vault with entity and relation extraction.
    
    Features:
    - Recursive markdown file discovery
    - Wiki-link [[]] parsing
    - Entity extraction per note (spaCy + optional LLM)
    - Relation extraction per note
    - Progress tracking
    
    v3.0 Features:
    - LLM-powered deep understanding (Gemma 3)
    - Triplet extraction for Graph RAG
    - Interest profile building
    - Decision pattern detection
    """
    
    def __init__(self, graph, spacy_model, llm_extractor=None, triplet_extractor=None):
        """
        Initialize Obsidian ingestor.
        
        Args:
            graph: PersonalGraph instance
            spacy_model: Pre-loaded spaCy model
            llm_extractor: Optional LLMExtractor for deep understanding
            triplet_extractor: Optional TripletExtractor for Graph RAG
        """
        super().__init__(graph, spacy_model)
        self.llm_extractor = llm_extractor
        self.triplet_extractor = triplet_extractor
        self.interest_profile = defaultdict(float)  # Track user interests
        self.decision_patterns = []  # Historical decision patterns
        
    def ingest(self, directory: str, use_llm: bool = False, sample_size: Optional[int] = None):
        """
        Ingest Obsidian vault with entity and relation extraction.
        
        Args:
            directory: Path to Obsidian vault directory
            use_llm: Whether to use LLM for advanced extraction (slower but deeper)
            sample_size: Optional limit on number of files to process (for testing)
        """
        if not os.path.exists(directory):
            self.log(f"❌ Path not found: {directory}")
            return
        
        md_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
        
        # Apply sample size limit if specified
        if sample_size and sample_size < len(md_files):
            md_files = md_files[:sample_size]
            self.log(f"📂 Processing sample of {len(md_files)} notes (use_llm={use_llm})")
        else:
            self.log(f"📂 Found {len(md_files)} notes in Obsidian vault (use_llm={use_llm})")
        
        progress_bar = st.progress(0)
        
        for i, filepath in enumerate(md_files):
            filename = os.path.basename(filepath).replace(".md", "")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add base note node
                self.graph.add_node(filename, NodeType.CONCEPT, content, 
                                  metadata={'source': 'obsidian', 'path': filepath})
                self.stats['notes'] += 1
                
                # === BASIC EXTRACTION (spaCy) ===
                entities = self.extract_entities(content)
                for ent_text, ent_type in entities:
                    ent_id = f"{ent_type.value}:{ent_text}"
                    self.graph.add_node(ent_id, ent_type, ent_text)
                    self.graph.add_edge(filename, ent_id, RelationType.MENTIONS)
                    self.stats['entities'] += 1
                
                # Extract relations
                relations = self.extract_relations(content, entities)
                for source, rel_type, target in relations:
                    # Create mini-nodes for relation endpoints if needed
                    src_id = f"phrase:{source}"
                    tgt_id = f"phrase:{target}"
                    
                    if src_id not in self.graph.graph:
                        self.graph.add_node(src_id, NodeType.CONCEPT, source)
                    if tgt_id not in self.graph.graph:
                        self.graph.add_node(tgt_id, NodeType.CONCEPT, target)
                    
                    self.graph.add_edge(src_id, tgt_id, rel_type)
                    self.stats['relations'] += 1
                
                # Obsidian [[wiki-links]]
                wikilink_pattern = r'\[\[([^\]]+)\]\]'
                for match in re.finditer(wikilink_pattern, content):
                    linked_note = match.group(1)
                    if linked_note in self.graph.graph:
                        self.graph.add_edge(filename, linked_note, RelationType.LINKS_TO)
                        self.stats['links'] += 1
                
                # === ADVANCED EXTRACTION (LLM) ===
                if use_llm and self.llm_extractor:
                    self._process_with_llm(filename, content)
                
            except Exception as e:
                self.log(f"⚠️ Error processing {filename}: {str(e)[:50]}")
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(md_files))
        
        progress_bar.empty()
        
        # Final summary
        if use_llm and self.llm_extractor:
            self._log_llm_summary()
        else:
            self.log(f"✅ Obsidian: {self.stats['notes']} notes, {self.stats['entities']} entities, {self.stats['relations']} relations")
    
    def _process_with_llm(self, filename: str, content: str):
        """
        Deep processing with LLM for rich understanding.
        
        Args:
            filename: Note filename
            content: Note content
        """
        # 1. Full note analysis
        analysis = self.llm_extractor.analyze_note(content, filename)
        
        if not analysis:
            return
        
        # 2. Add rich entities
        for entity in analysis.entities:
            ent_id = f"{entity.type}:{entity.text}"
            
            # Add with sentiment and importance metadata
            self.graph.add_node(
                ent_id, 
                NodeType(entity.type) if entity.type in NodeType.__members__.values() else NodeType.CONCEPT,
                entity.text,
                metadata={
                    'sentiment': entity.sentiment,
                    'importance': entity.importance,
                    'context': entity.context
                }
            )
            
            self.graph.add_edge(filename, ent_id, RelationType.MENTIONS)
            self.stats['llm_entities'] = self.stats.get('llm_entities', 0) + 1
            
            # Update interest profile
            if entity.importance and entity.importance > 0.5:
                self.interest_profile[entity.text] += entity.importance
        
        # 3. Add semantic relations
        for relation in analysis.relations:
            src_id = f"concept:{relation.source}"
            tgt_id = f"concept:{relation.target}"
            
            # Ensure nodes exist
            if src_id not in self.graph.graph:
                self.graph.add_node(src_id, NodeType.CONCEPT, relation.source)
            if tgt_id not in self.graph.graph:
                self.graph.add_node(tgt_id, NodeType.CONCEPT, relation.target)
            
            # Add edge with confidence and evidence
            rel_type = self._map_relation_type(relation.relation_type)
            self.graph.add_edge(
                src_id, tgt_id, rel_type,
                weight=relation.confidence,
                metadata={'evidence': relation.evidence}
            )
            self.stats['llm_relations'] = self.stats.get('llm_relations', 0) + 1
        
        # 4. Extract triplets for Graph RAG
        if self.triplet_extractor:
            triplets = self.triplet_extractor.extract_from_text(content)
            self.stats['triplets'] = self.stats.get('triplets', 0) + len(triplets)
        
        # 5. Detect decision patterns
        if analysis.decision_intent:
            decision_factors = self.llm_extractor.extract_decision_factors(content)
            if decision_factors.get('decision_question'):
                self.decision_patterns.append({
                    'note': filename,
                    'question': decision_factors['decision_question'],
                    'options': decision_factors['options'],
                    'factors': decision_factors['factors'],
                    'risks': decision_factors['risks'],
                    'sentiment': decision_factors['sentiment']
                })
                self.stats['decisions_found'] = self.stats.get('decisions_found', 0) + 1
    
    def _map_relation_type(self, llm_relation: str) -> RelationType:
        """Map LLM relation types to our RelationType enum."""
        mapping = {
            'causes': RelationType.CAUSES,
            'enables': RelationType.SUPPORTS,
            'conflicts_with': RelationType.CONTRADICTS,
            'requires': RelationType.RELATED_TO,
            'supports': RelationType.SUPPORTS,
        }
        return mapping.get(llm_relation, RelationType.RELATED_TO)
    
    def _log_llm_summary(self):
        """Log comprehensive summary with LLM stats."""
        summary = f"""
✅ Obsidian Ingestion Complete!
   📄 Notes: {self.stats['notes']}
   
   Basic Extraction (spaCy):
   - Entities: {self.stats['entities']}
   - Relations: {self.stats['relations']}
   - Wiki-links: {self.stats['links']}
   
   Advanced Extraction (LLM):
   - Rich Entities: {self.stats.get('llm_entities', 0)}
   - Semantic Relations: {self.stats.get('llm_relations', 0)}
   - Triplets: {self.stats.get('triplets', 0)}
   - Decisions Found: {self.stats.get('decisions_found', 0)}
   
   Top 5 Interests Detected:
"""
        # Show top interests
        top_interests = sorted(self.interest_profile.items(), key=lambda x: x[1], reverse=True)[:5]
        for interest, score in top_interests:
            summary += f"   - {interest}: {score:.2f}\n"
        
        self.log(summary)
    
    def get_interest_profile(self) -> dict:
        """Get the built interest profile."""
        return dict(self.interest_profile)
    
    def get_decision_patterns(self) -> list:
        """Get detected decision patterns."""
        return self.decision_patterns

