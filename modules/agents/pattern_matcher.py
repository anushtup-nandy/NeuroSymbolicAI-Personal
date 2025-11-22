"""
Historical Pattern Matcher
===========================
Match current decisions to historical patterns from notes.

Suggests factor weights based on what the user has valued in the past.
"""

from typing import List, Dict, Optional
from collections import defaultdict, Counter
import re


class HistoricalPatternMatcher:
    """
    Match current decision to historical patterns.
    
    Learns from:
    - Past decisions found in Obsidian notes
    - Factor frequencies
    - Weight patterns (what user emphasizes)
    - Outcome sentiments
    
    Suggests:
    - Factor weights based on historical emphasis
    - Similar past decisions for reference
    - Common risks/considerations
    """
    
    def __init__(self, graph=None, decision_patterns: Optional[List[Dict]] = None):
        """
        Initialize pattern matcher.
        
        Args:
            graph: PersonalGraph instance (optional)
            decision_patterns: List of decision patterns from ingestion
        """
        self.graph = graph
        self.decision_patterns = decision_patterns or []
        self._build_pattern_index()
        
    def _build_pattern_index(self):
        """Build indices for fast pattern matching."""
        self.factor_frequency = Counter()
        self.factor_cooccurrence = defaultdict(Counter)
        self.decision_types = defaultdict(list)
        
        for pattern in self.decision_patterns:
            factors = pattern.get('factors', [])
            decision_q = pattern.get('question', '').lower()
            
            # Count factor frequencies
            for factor in factors:
                self.factor_frequency[factor] += 1
                
                # Track co-occurrence
                for other_factor in factors:
                    if factor != other_factor:
                        self.factor_cooccurrence[factor][other_factor] += 1
            
            # Categorize by decision type
            decision_type = self._infer_decision_type(decision_q)
            self.decision_types[decision_type].append(pattern)
    
    def _infer_decision_type(self, question: str) -> str:
        """
        Infer decision type from question.
        
        Args:
            question: Decision question text
            
        Returns:
            Decision type category
        """
        question_lower = question.lower()
        
        # Simple keyword matching (can be enhanced)
        if any(word in question_lower for word in ['job', 'career', 'work', 'position']):
            return 'career'
        elif any(word in question_lower for word in ['move', 'relocate', 'location', 'city', 'country']):
            return 'location'
        elif any(word in question_lower for word in ['buy', 'purchase', 'invest']):
            return 'purchase'
        elif any(word in question_lower for word in ['learn', 'study', 'course', 'education']):
            return 'education'
        else:
            return 'general'
    
    def suggest_weights(self, 
                       factors: List[str],
                       decision_question: str = "",
                       min_confidence: float = 0.3) -> Dict[str, float]:
        """
        Suggest factor weights based on historical patterns.
        
        Args:
            factors: List of factor names for current decision
            decision_question: Optional decision question for context
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict of factor_name -> suggested_weight
        """
        if not self.decision_patterns:
            # No history - use uniform weights
            weight = 1.0 / len(factors) if factors else 0.0
            return {f: weight for f in factors}
        
        # Find similar past decisions
        decision_type = self._infer_decision_type(decision_question)
        similar_decisions = self.decision_types.get(decision_type, [])
        
        if not similar_decisions:
            # Fall back to all decisions
            similar_decisions = self.decision_patterns
        
        # Aggregate weights from similar decisions
        weight_scores = defaultdict(float)
        
        for decision in similar_decisions:
            decision_factors = decision.get('factors', [])
            
            # Score based on factor overlap and frequency
            for factor in factors:
                factor_lower = factor.lower()
                
                # Exact match
                if factor_lower in [f.lower() for f in decision_factors]:
                    weight_scores[factor] += 2.0
                
                # Partial match (substring)
                for hist_factor in decision_factors:
                    if factor_lower in hist_factor.lower() or hist_factor.lower() in factor_lower:
                        weight_scores[factor] += 1.0
                
                # Frequency boost
                weight_scores[factor] += self.factor_frequency.get(factor, 0) * 0.1
        
        # Normalize to sum to 1.0
        total_score = sum(weight_scores.values())
        
        if total_score == 0:
            # No matches - uniform weights
            weight = 1.0 / len(factors) if factors else 0.0
            return {f: weight for f in factors}
        
        suggested_weights = {
            factor: weight_scores[factor] / total_score
            for factor in factors
        }
        
        return suggested_weights
    
    def find_similar_decisions(self, 
                               question: str,
                               factors: List[str],
                               top_k: int = 3) -> List[Dict]:
        """
        Find most similar historical decisions.
        
        Args:
            question: Current decision question
            factors: Current decision factors
            top_k: Number of similar decisions to return
            
        Returns:
            List of similar decision patterns
        """
        if not self.decision_patterns:
            return []
        
        decision_type = self._infer_decision_type(question)
        candidates = self.decision_types.get(decision_type, self.decision_patterns)
        
        # Score each candidate by factor overlap
        scored_decisions = []
        
        for decision in candidates:
            hist_factors = set(f.lower() for f in decision.get('factors', []))
            curr_factors = set(f.lower() for f in factors)
            
            # Jaccard similarity
            intersection = len(hist_factors & curr_factors)
            union = len(hist_factors | curr_factors)
            
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > 0:
                scored_decisions.append((similarity, decision))
        
        # Sort by similarity and return top_k
        scored_decisions.sort(key=lambda x: x[0], reverse=True)
        
        return [decision for _, decision in scored_decisions[:top_k]]
    
    def get_common_risks(self, decision_type: str = "general", top_k: int = 5) -> List[str]:
        """
        Get common risks from historical decisions.
        
        Args:
            decision_type: Type of decision
            top_k: Number of risks to return
            
        Returns:
            List of common risks
        """
        decisions = self.decision_types.get(decision_type, self.decision_patterns)
        
        risk_counter = Counter()
        
        for decision in decisions:
            risks = decision.get('risks', [])
            for risk in risks:
                risk_counter[risk] += 1
        
        return [risk for risk, count in risk_counter.most_common(top_k)]
    
    def get_stats(self) -> Dict:
        """Get pattern matcher statistics."""
        return {
            'total_patterns': len(self.decision_patterns),
            'decision_types': dict(Counter(self._infer_decision_type(p.get('question', '')) for p in self.decision_patterns)),
            'most_common_factors': self.factor_frequency.most_common(10),
            'unique_factors': len(self.factor_frequency)
        }
