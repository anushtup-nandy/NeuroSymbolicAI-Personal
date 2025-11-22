"""
LLM-Powered Entity and Relation Extraction
===========================================
Uses Gemma 3 (4b) with function calling for deep understanding of Obsidian notes.

Features:
- Structured output using Pydantic schemas
- Rich entity extraction (domain-aware)
- Implicit relation discovery
- Sentiment analysis
- Contradiction detection
"""

import json
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from collections import defaultdict


# ========================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# ========================================

class Entity(BaseModel):
    """Extracted entity with rich metadata."""
    text: str = Field(..., description="The entity text")
    type: str = Field(..., description="Entity type: person, org, concept, decision, value, risk, location, event, etc.")
    context: str = Field(..., description="Surrounding sentence or context")
    sentiment: Optional[float] = Field(None, description="Sentiment score from -1 (negative) to 1 (positive)")
    importance: Optional[float] = Field(None, description="Importance score from 0 to 1")


class Relation(BaseModel):
    """Extracted relationship between entities."""
    source: str = Field(..., description="Source entity")
    relation_type: str = Field(..., description="Type: causes, enables, conflicts_with, requires, supports, etc.")
    target: str = Field(..., description="Target entity")
    confidence: float = Field(0.8, description="Confidence score from 0 to 1")
    evidence: str = Field(..., description="Supporting text snippet")


class Triplet(BaseModel):
    """Knowledge triplet (Subject, Predicate, Object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8


class NoteAnalysis(BaseModel):
    """Complete analysis of an Obsidian note."""
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    main_topics: List[str] = Field(default_factory=list)
    sentiment_overall: float = Field(0.0, description="Overall sentiment from -1 to 1")
    decision_intent: Optional[str] = Field(None, description="If note discusses a decision, describe it")
    contradictions: List[str] = Field(default_factory=list)


# ========================================
# LLM EXTRACTOR
# ========================================

class LLMExtractor:
    """
    Extract rich information from text using Gemma 3 with function calling.
    
    Uses Ollama's JSON mode to get structured output matching Pydantic schemas.
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model: str = "gemma3:4b",
                 cache_enabled: bool = True):
        """
        Initialize LLM extractor.
        
        Args:
            ollama_url: Ollama server URL
            model: Model name (default: gemma3:4b)
            cache_enabled: Whether to cache results
        """
        self.ollama_url = ollama_url
        self.model = model
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, Any] = {}
        self.stats = defaultdict(int)
        
    def analyze_note(self, content: str, note_title: str) -> Optional[NoteAnalysis]:
        """
        Full analysis of a note with structured output.
        
        Args:
            content: Note content (markdown)
            note_title: Note title/filename
            
        Returns:
            NoteAnalysis object or None if extraction fails
        """
        # Check cache
        cache_key = f"{note_title}:{hash(content)}"
        if self.cache_enabled and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        # Build prompt for comprehensive analysis
        prompt = self._build_analysis_prompt(content, note_title)
        
        try:
            # Call Ollama with JSON schema
            response_json = self._call_ollama_json(prompt)
            
            # Parse and validate with Pydantic
            analysis = NoteAnalysis.model_validate(response_json)
            
            # Cache result
            if self.cache_enabled:
                self.cache[cache_key] = analysis
            
            self.stats['successful_extractions'] += 1
            return analysis
            
        except Exception as e:
            self.stats['failed_extractions'] += 1
            print(f"⚠️ LLM extraction failed for {note_title}: {str(e)[:100]}")
            return None
    
    def extract_triplets(self, content: str, max_triplets: int = 20) -> List[Triplet]:
        """
        Extract (Subject, Predicate, Object) triplets for Graph RAG.
        
        Example: "USA requires visa" → Triplet(subject="USA", predicate="requires", object="visa")
        
        Args:
            content: Text to extract from
            max_triplets: Maximum number of triplets to extract
            
        Returns:
            List of Triplet objects
        """
        prompt = f"""Extract knowledge triplets from this text.
Each triplet should be (Subject, Predicate, Object).

Focus on:
- Causal relationships (X causes Y, X leads to Y)
- Requirements (X requires Y, X needs Y)
- Conflicts (X conflicts with Y, X opposes Y)
- Support (X supports Y, X enables Y)

Text:
{content[:2000]}

Output as JSON array with schema:
[{{"subject": "entity1", "predicate": "relation", "object": "entity2", "confidence": 0.9}}]

Maximum {max_triplets} triplets.
"""
        
        try:
            response_json = self._call_ollama_json(prompt)
            
            # Handle both dict and list responses
            if isinstance(response_json, dict) and 'triplets' in response_json:
                triplets_data = response_json['triplets']
            elif isinstance(response_json, list):
                triplets_data = response_json
            else:
                triplets_data = []
            
            triplets = [Triplet(**t) for t in triplets_data[:max_triplets]]
            self.stats['triplets_extracted'] += len(triplets)
            return triplets
            
        except Exception as e:
            print(f"⚠️ Triplet extraction failed: {str(e)[:100]}")
            return []
    
    def extract_decision_factors(self, content: str) -> Dict[str, Any]:
        """
        Extract decision-related information from text.
        
        Args:
            content: Text that may contain decision information
            
        Returns:
            Dict with options, factors, sentiments, risks
        """
        prompt = f"""Analyze this text for decision-making content.

Text:
{content[:1500]}

Extract and output as JSON:
{{
    "decision_question": "What decision is being considered?",
    "options": ["option1", "option2"],
    "factors": ["factor1", "factor2"],
    "risks": ["risk1", "risk2"],
    "sentiment": 0.5
}}

If no decision is discussed, return empty arrays.
"""
        
        try:
            result = self._call_ollama_json(prompt)
            self.stats['decisions_extracted'] += 1
            return result
        except Exception as e:
            print(f"⚠️ Decision extraction failed: {str(e)[:100]}")
            return {
                "decision_question": None,
                "options": [],
                "factors": [],
                "risks": [],
                "sentiment": 0.0
            }
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _build_analysis_prompt(self, content: str, note_title: str) -> str:
        """Build comprehensive analysis prompt."""
        # Truncate content if too long
        max_content_length = 3000
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "\n... (content truncated)"
        
        prompt = f"""Analyze this Obsidian note and extract structured information.

Note Title: {note_title}

Content:
{truncated_content}

Extract:
1. **Entities**: People, organizations, concepts, decisions, values, risks, locations, events
   - Include sentiment (-1 to 1) and importance (0 to 1) for each
2. **Relations**: Relationships between entities (causes, enables, conflicts_with, requires, supports)
3. **Main Topics**: 3-5 key topics discussed
4. **Overall Sentiment**: General tone (-1 negative to 1 positive)
5. **Decision Intent**: If discussing a decision, describe it
6. **Contradictions**: Any conflicting statements

Output as JSON matching this schema:
{{
    "entities": [{{"text": "entity", "type": "person|org|concept|decision|value|risk|location|event", "context": "surrounding text", "sentiment": 0.5, "importance": 0.8}}],
    "relations": [{{"source": "entity1", "relation_type": "causes|enables|conflicts_with|requires|supports", "target": "entity2", "confidence": 0.9, "evidence": "supporting text"}}],
    "main_topics": ["topic1", "topic2"],
    "sentiment_overall": 0.5,
    "decision_intent": "description or null",
    "contradictions": ["contradiction1"]
}}

Be thorough but concise. Focus on extracting meaningful information.
"""
        return prompt
    
    def _call_ollama_json(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Call Ollama with JSON mode enabled.
        
        Args:
            prompt: Prompt text
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Parsed JSON response
        """
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",  # Force JSON output
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048  # Max tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '{}')
        
        # Parse JSON
        return json.loads(response_text)
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return dict(self.stats)
    
    def clear_cache(self):
        """Clear the extraction cache."""
        self.cache.clear()
        self.stats['cache_cleared'] = self.stats.get('cache_cleared', 0) + 1
