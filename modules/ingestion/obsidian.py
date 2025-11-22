"""
Obsidian Vault Ingestion
=========================
Ingest Obsidian markdown files with entity and relation extraction.
"""

import os
import glob
import re
import streamlit as st

from modules.core.types import NodeType, RelationType
from modules.ingestion.base import BaseIngestor


class ObsidianIngestor(BaseIngestor):
    """
    Ingest Obsidian vault with entity and relation extraction.
    
    Features:
    - Recursive markdown file discovery
    - Wiki-link [[]] parsing
    - Entity extraction per note
    - Relation extraction per note
    - Progress tracking
    """
    
    def ingest(self, directory: str, use_llm: bool = False):
        """
        Ingest Obsidian vault with entity and relation extraction.
        
        Args:
            directory: Path to Obsidian vault directory
            use_llm: Whether to use LLM for advanced extraction (slower)
        """
        if not os.path.exists(directory):
            self.log(f"❌ Path not found: {directory}")
            return
        
        md_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
        self.log(f"📂 Found {len(md_files)} notes in Obsidian vault")
        
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
                
                # Extract entities
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
                
            except Exception as e:
                self.log(f"⚠️ Error processing {filename}: {str(e)[:50]}")
            
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(md_files))
        
        progress_bar.empty()
        self.log(f"✅ Obsidian: {self.stats['notes']} notes, {self.stats['entities']} entities, {self.stats['relations']} relations")
