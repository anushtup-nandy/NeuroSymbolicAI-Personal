"""
Natural Language Decision Parser
=================================
Parse natural language decision queries into structured decision matrices.

Transform: "Should I take Job A or Job B?" → Auto-populated decision matrix

Features:
- Extract options from natural language
- Identify decision factors (explicit and implicit)
- Auto-score based on sentiment analysis
- Leverage knowledge graph for context
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from collections import defaultdict

from modules.ingestion.llm_extractor import LLMExtractor
from modules.graph import analytics


# ========================================
# PYDANTIC SCHEMAS
# ========================================

class DecisionOption(BaseModel):
    """Parsed decision option."""
    name: str = Field(..., description="Option name")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Attributes like salary, location, etc.")
    mentioned_pros: List[str] = Field(default_factory=list, description="Positive aspects mentioned")
    mentioned_cons: List[str] = Field(default_factory=list, description="Negative aspects mentioned")


class DecisionFactor(BaseModel):
    """Parsed decision factor."""
    name: str = Field(..., description="Factor name")
    weight: float = Field(..., description="Weight from 0 to 1")
    type: str = Field("benefit", description="'benefit' or 'cost'")
    importance_from_text: Optional[float] = Field(None, description="Importance mentioned in text (0-1)")


class ParsedDecision(BaseModel):
    """Complete parsed decision."""
    question: str = Field(..., description="Clear decision question")
    options: List[DecisionOption] = Field(default_factory=list)
    factors: List[DecisionFactor] = Field(default_factory=list)
    initial_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="option -> {factor: score}")
    context: str = Field("", description="Original text")
    confidence: float = Field(0.8, description="Parser confidence 0-1")


# ========================================
# DECISION PARSER
# ========================================

class DecisionParser:
    """
    Parse natural language into structured decision format.
    
    Example Input:
        "Should I move to USA or stay in India? USA has better career growth
         but India is closer to family. I care most about long-term growth."
    
    Example Output:
        ParsedDecision(
            question="Should I move to USA or stay in India?",
            options=[
                DecisionOption(name="USA", mentioned_pros=["career growth"], mentioned_cons=["far from family"]),
                DecisionOption(name="India", mentioned_pros=["family proximity"], mentioned_cons=["limited growth"])
            ],
            factors=[
                DecisionFactor(name="career_growth", weight=0.5),
                DecisionFactor(name="family_proximity", weight=0.3),
                DecisionFactor(name="quality_of_life", weight=0.2)
            ],
            initial_scores={
                "USA": {"career_growth": 9, "family_proximity": 3, "quality_of_life": 7},
                "India": {"career_growth": 6, "family_proximity": 9, "quality_of_life": 6}
            }
        )
    """
    
    def __init__(self, llm_extractor: LLMExtractor, graph=None):
        """
        Initialize decision parser.
        
        Args:
            llm_extractor: LLM extractor instance
            graph: Optional PersonalGraph for context
        """
        self.llm_extractor = llm_extractor
        self.graph = graph
        self.parse_cache = {}
        
    def parse(self, text: str, use_graph_context: bool = True) -> ParsedDecision:
        """
        Parse natural language decision query.
        
        Args:
            text: Natural language decision description
            use_graph_context: Whether to use knowledge graph for context
            
        Returns:
            ParsedDecision object with auto-populated fields
        """
        # Check cache
        cache_key = hash(text)
        if cache_key in self.parse_cache:
            return self.parse_cache[cache_key]
        
        # 1. Get graph context if available
        context_notes = []
        if use_graph_context and self.graph:
            context_notes = self._get_relevant_context(text)
        
        # 2. Build LLM prompt with schema
        prompt = self._build_parsing_prompt(text, context_notes)
        
        # 3. Call LLM with structured output
        try:
            response = self.llm_extractor._call_ollama_json(prompt, temperature=0.3)
            parsed = ParsedDecision.model_validate(response)
        except Exception as e:
            print(f"⚠️ Parsing failed: {e}")
            # Return minimal valid structure
            return ParsedDecision(
                question=text[:100],
                options=[],
                factors=[],
                initial_scores={},
                context=text,
                confidence=0.0
            )
        
        # 4. Auto-score if scores not provided
        if not parsed.initial_scores or len(parsed.initial_scores) == 0:
            parsed.initial_scores = self._generate_scores(parsed, text)
        
        # 5. Normalize weights
        parsed.factors = self._normalize_weights(parsed.factors)
        
        # Cache result
        self.parse_cache[cache_key] = parsed
        
        return parsed
    
    def _build_parsing_prompt(self, text: str, context_notes: List[str]) -> str:
        """Build prompt for decision parsing."""
        context_str = ""
        if context_notes:
            context_str = "\n\nRelevant context from user's notes:\n" + "\n".join(f"- {note}" for note in context_notes[:3])
        
        return f"""Parse this decision query into structured format for decision analysis.

User's Query:
{text}
{context_str}

Extract and structure:

1. **Question**: Rephrase as a clear decision question
2. **Options**: All alternatives being considered
   - name: Clear option name
   - attributes: Specific attributes mentioned (salary: 120000, location: "remote", etc.)
   - mentioned_pros: Positive aspects mentioned
   - mentioned_cons: Negative aspects mentioned

3. **Factors**: Decision criteria (both explicit and reasonably inferred)
   - name: Factor name (use snake_case)
   - weight: Initial weight 0-1 (higher if emphasized in text, must sum to 1.0)
   - type: "benefit" (more is better) or "cost" (less is better)

4. **Initial Scores**: Best guess scores 0-10 based on:
   - Explicit comparisons in text
   - Sentiment analysis (positive mentions = higher score)
   - Attributes (higher salary = higher score on salary factor)

Output as JSON matching this schema:
{{
    "question": "Clear, specific decision question",
    "options": [
        {{
            "name": "Option A",
            "attributes": {{"key": "value"}},
            "mentioned_pros": ["pro1", "pro2"],
            "mentioned_cons": ["con1"]
        }}
    ],
    "factors": [
        {{"name": "factor_name", "weight": 0.4, "type": "benefit"}},
        {{"name": "another_factor", "weight": 0.3, "type": "benefit"}},
        {{"name": "cost_factor", "weight": 0.3, "type": "cost"}}
    ],
    "initial_scores": {{
        "Option A": {{"factor_name": 8, "another_factor": 6, "cost_factor": 4}}
    }},
    "context": "{text[:200]}...",
    "confidence": 0.8
}}

Guidelines:
- Be comprehensive but focused on what's mentioned or clearly implied
- Weights must sum to 1.0
- Scores are 0-10 (0=worst, 10=best)
- Use snake_case for factor names
- Infer unstated but obvious factors (e.g., cost is often implicit)
"""
    
    def _get_relevant_context(self, text: str, top_k: int = 5) -> List[str]:
        """
        Get relevant context from knowledge graph.
        
        Args:
            text: Query text
            top_k: Number of relevant notes to retrieve
            
        Returns:
            List of relevant note snippets
        """
        if not self.graph:
            return []
        
        try:
            # Semantic search in knowledge graph
            results = analytics.semantic_search(self.graph, text, top_k=top_k)
            
            context_notes = []
            for node_id, similarity, data in results:
                if similarity > 0.5:  # Only include relevant matches
                    content = data.get('content', '')
                    # Take first 200 chars as context
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    context_notes.append(f"{node_id}: {snippet}")
            
            return context_notes
        except Exception as e:
            print(f"⚠️ Context retrieval failed: {e}")
            return []
    
    def _generate_scores(self, parsed: ParsedDecision, text: str) -> Dict[str, Dict[str, float]]:
        """
        Auto-generate scores using sentiment analysis and mentioned attributes.
        
        Args:
            parsed: ParsedDecision object
            text: Original text for sentiment analysis
            
        Returns:
            Dict of scores: option_name -> {factor_name: score}
        """
        scores = {}
        text_lower = text.lower()
        
        for option in parsed.options:
            scores[option.name] = {}
            
            for factor in parsed.factors:
                # Start with neutral score
                score = 5.0
                
                factor_name = factor.name.replace('_', ' ')
                option_name_lower = option.name.lower()
                
                # Check if this option is mentioned positively with this factor
                for pro in option.mentioned_pros:
                    if factor_name.lower() in pro.lower():
                        score += 3.0
                
                # Check if mentioned negatively
                for con in option.mentioned_cons:
                    if factor_name.lower() in con.lower():
                        score -= 3.0
                
                # Check attributes (e.g., salary value)
                if factor.name in option.attributes:
                    attr_value = option.attributes[factor.name]
                    # Normalize numeric attributes to 0-10 scale (simple heuristic)
                    if isinstance(attr_value, (int, float)):
                        # Assume reasonable range and map to 0-10
                        # This is a rough heuristic - could be improved
                        score = min(10, max(0, attr_value / 10000))  # Example: salary/10k
                
                # Clamp to 0-10
                score = max(0, min(10, score))
                
                scores[option.name][factor.name] = round(score, 1)
        
        return scores
    
    def _normalize_weights(self, factors: List[DecisionFactor]) -> List[DecisionFactor]:
        """
        Normalize factor weights to sum to 1.0.
        
        Args:
            factors: List of DecisionFactor objects
            
        Returns:
            List of DecisionFactor objects with normalized weights
        """
        if not factors:
            return factors
        
        total_weight = sum(f.weight for f in factors)
        
        if total_weight == 0:
            # Assign uniform weights
            uniform_weight = 1.0 / len(factors)
            for f in factors:
                f.weight = uniform_weight
        elif abs(total_weight - 1.0) > 0.01:
            # Normalize to sum to 1.0
            for f in factors:
                f.weight = f.weight / total_weight
        
        return factors
    
    def parse_multiple(self, texts: List[str]) -> List[ParsedDecision]:
        """
        Parse multiple decision queries.
        
        Args:
            texts: List of decision query texts
            
        Returns:
            List of ParsedDecision objects
        """
        return [self.parse(text) for text in texts]
