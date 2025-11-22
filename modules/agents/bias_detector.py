"""
Bias Detector ("Devil's Advocate")
===================================
Detect cognitive biases and inconsistencies in decision-making.

Compares current decision against:
- Historical weight patterns
- Known preferences from notes
- Warning signs (overconfidence, sunk cost, etc.)
"""

from typing import List, Dict, Optional
from collections import defaultdict
import statistics


class BiasWarning:
    """Represents a detected bias or inconsistency."""
    
    def __init__(self, 
                 bias_type: str,
                 severity: str,  # 'low', 'medium', 'high'
                 message: str,
                 evidence: str = "",
                 suggestion: str = ""):
        self.bias_type = bias_type
        self.severity = severity
        self.message = message
        self.evidence = evidence
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'bias_type': self.bias_type,
            'severity': self. severity,
            'message': self.message,
            'evidence': self.evidence,
            'suggestion': self.suggestion
        }


class BiasDetector:
    """
    Detect cognitive biases and inconsistencies.
    
    Detects:
    1. **Weight Inconsistency**: Current weights differ from historical patterns
    2. **Overconfidence**: All scores are extreme (no uncertainty)
    3. **Interest Mismatch**: Factors don't align with user's interests
    4. **Confirmation Bias**: Scores suspiciously favor one option
    5. **Sunk Cost**: High weight on factors related to past investment
    6. **Anchoring**: First option scored much higher
    """
    
    def __init__(self):
        """Initialize bias detector."""
        self.warnings_generated = 0
        
    def detect_biases(self,
                      parsed_decision,  # ParsedDecision object
                      historical_patterns: Optional[List[Dict]] = None,
                      interest_profile: Optional[Dict[str, float]] = None,
                      suggested_weights: Optional[Dict[str, float]] = None) -> List[BiasWarning]:
        """
        Detect potential biases in the decision.
        
        Args:
            parsed_decision: ParsedDecision object
            historical_patterns: Historical decision patterns
            interest_profile: User's interest profile from notes
            suggested_weights: Historically suggested weights
            
        Returns:
            List of BiasWarning objects
        """
        warnings = []
        
        # 1. Weight inconsistency
        if suggested_weights and len(parsed_decision.factors) > 0:
            weight_warnings = self._check_weight_inconsistency(
                parsed_decision.factors,
                suggested_weights
            )
            warnings.extend(weight_warnings)
        
        # 2. Overconfidence
        if parsed_decision.initial_scores:
            overconfidence = self._check_overconfidence(parsed_decision.initial_scores)
            if overconfidence:
                warnings.append(overconfidence)
        
        # 3. Interest mismatch
        if interest_profile and len(parsed_decision.factors) > 0:
            interest_warnings = self._check_interest_mismatch(
                parsed_decision.factors,
                interest_profile
            )
            warnings.extend(interest_warnings)
        
        # 4. Confirmation bias
        if parsed_decision.initial_scores and len(parsed_decision.options) > 1:
            confirmation = self._check_confirmation_bias(
                parsed_decision.initial_scores,
                parsed_decision.options
            )
            if confirmation:
                warnings.append(confirmation)
        
        # 5. Anchoring bias
        if parsed_decision.initial_scores and len(parsed_decision.options) > 1:
            anchoring = self._check_anchoring_bias(
                parsed_decision.initial_scores,
                parsed_decision.options
            )
            if anchoring:
                warnings.append(anchoring)
        
        self.warnings_generated = len(warnings)
        
        return warnings
    
    def _check_weight_inconsistency(self,
                                   factors: List,
                                   suggested_weights: Dict[str, float],
                                   threshold: float = 0.3) -> List[BiasWarning]:
        """
        Check if weights differ significantly from historical patterns.
        
        Args:
            factors: List of DecisionFactor objects
            suggested_weights: Historically suggested weights
            threshold: Difference threshold to trigger warning
            
        Returns:
            List of warnings
        """
        warnings = []
        
        for factor in factors:
            suggested = suggested_weights.get(factor.name, None)
            
            if suggested is None:
                continue
            
            difference = abs(factor.weight - suggested)
            
            if difference > threshold:
                # Significant deviation
                if factor.weight > suggested:
                    direction = "higher"
                    verb = "overemphasizing"
                else:
                    direction = "lower"
                    verb = "underemphasizing"
                
                warning = BiasWarning(
                    bias_type="weight_inconsistency",
                    severity="medium" if difference > 0.4 else "low",
                    message=f"⚠️ You weighted '{factor.name}' {direction} than historical patterns suggest",
                    evidence=f"Current: {factor.weight:.2f}, Historical: {suggested:.2f} (Δ{difference:.2f})",
                    suggestion=f"Based on your past decisions, you typically value '{factor.name}' at {suggested:.2f}. Consider if you're {verb} this factor."
                )
                warnings.append(warning)
        
        return warnings
    
    def _check_overconfidence(self,
                             scores: Dict[str, Dict[str, float]],
                             extreme_threshold: float = 8.5) -> Optional[BiasWarning]:
        """
        Check if all scores are extreme (very high or very low).
        
        Indicates overconfidence or insufficient uncertainty consideration.
        
        Args:
            scores: option -> {factor: score}
            extreme_threshold: Threshold for "extreme" score
            
        Returns:
            BiasWarning if overconfidence detected, None otherwise
        """
        all_scores = []
        for option_scores in scores.values():
            all_scores.extend(option_scores.values())
        
        if not all_scores:
            return None
        
        # Count extreme scores
        extreme_count = sum(1 for score in all_scores if score >= extreme_threshold or score <= (10 - extreme_threshold))
        extreme_ratio = extreme_count / len(all_scores)
        
        if extreme_ratio > 0.7:  # More than 70% are extreme
            return BiasWarning(
                bias_type="overconfidence",
                severity="medium",
                message="⚠️ Most scores are extreme (very high or very low)",
                evidence=f"{extreme_ratio*100:.0f}% of scores are > {extreme_threshold} or < {10-extreme_threshold}",
                suggestion="Consider if you're being overconfident. Real decisions usually have more uncertainty and nuance."
            )
        
        return None
    
    def _check_interest_mismatch(self,
                                factors: List,
                                interest_profile: Dict[str, float],
                                top_k: int = 5) -> List[BiasWarning]:
        """
        Check if decision factors align with user's known interests.
        
        Args:
            factors: List of DecisionFactor objects
            interest_profile: User's interest profile
            top_k: Number of top interests to consider
            
        Returns:
            List of warnings
        """
        warnings = []
        
        if not interest_profile:
            return warnings
        
        # Get top interests
        top_interests = sorted(interest_profile.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_interest_names = [name.lower() for name, score in top_interests]
        
        # Check if high-weight factors align with interests
        high_weight_factors = [f for f in factors if f.weight > 0.3]
        
        for factor in high_weight_factors:
            factor_lower = factor.name.replace('_', ' ').lower()
            
            # Check if factor matches any top interest
            matches_interest = any(
                interest in factor_lower or factor_lower in interest
                for interest in top_interest_names
            )
            
            if not matches_interest:
                # High-weight factor doesn't match known interests
                top_3_interests = ", ".join([name for name, _ in top_interests[:3]])
                
                warning = BiasWarning(
                    bias_type="interest_mismatch",
                    severity="low",
                    message=f"ℹ️ Factor '{factor.name}' has high weight but doesn't match your known interests",
                    evidence=f"Your top interests from notes: {top_3_interests}",
                    suggestion=f"Based on your notes, you typically care more about: {top_3_interests}. Is '{factor.name}' really that important?"
                )
                warnings.append(warning)
        
        return warnings
    
    def _check_confirmation_bias(self,
                                scores: Dict[str, Dict[str, float]],
                                options: List,
                                threshold: float = 3.0) -> Optional[BiasWarning]:
        """
        Check if scores suspiciously favor one option.
        
        Args:
            scores: option -> {factor: score}
            options: List of DecisionOption objects
            threshold: Average score difference threshold
            
        Returns:
            BiasWarning if confirmation bias detected
        """
        if len(scores) < 2:
            return None
        
        # Calculate average score per option
        avg_scores = {}
        for option_name, factor_scores in scores.items():
            if factor_scores:
                avg_scores[option_name] = statistics.mean(factor_scores.values())
        
        if len(avg_scores) < 2:
            return None
        
        # Find max and min
        max_option = max(avg_scores.items(), key=lambda x: x[1])
        min_option = min(avg_scores.items(), key=lambda x: x[1])
        
        difference = max_option[1] - min_option[1]
        
        if difference > threshold:
            return BiasWarning(
                bias_type="confirmation_bias",
                severity="medium",
                message=f"⚠️ '{max_option[0]}' scores suspiciously higher than '{min_option[0]}'",
                evidence=f"Average scores: {max_option[0]}={max_option[1]:.1f}, {min_option[0]}={min_option[1]:.1f} (Δ{difference:.1f})",
                suggestion="You might be unconsciously favoring one option. Try to be more objective when scoring."
            )
        
        return None
    
    def _check_anchoring_bias(self,
                             scores: Dict[str, Dict[str, float]],
                             options: List,
                             threshold: float = 2.5) -> Optional[BiasWarning]:
        """
        Check if first option is scored significantly higher (anchoring).
        
        Args:
            scores: option -> {factor: score}
            options: List of DecisionOption objects
            threshold: Difference threshold
            
        Returns:
            BiasWarning if anchoring detected
        """
        if len(options) < 2:
            return None
        
        first_option_name = options[0].name
        
        if first_option_name not in scores or not scores[first_option_name]:
            return None
        
        first_avg = statistics.mean(scores[first_option_name].values())
        
        other_avgs = []
        for option in options[1:]:
            if option.name in scores and scores[option.name]:
                other_avgs.append(statistics.mean(scores[option.name].values()))
        
        if not other_avgs:
            return None
        
        avg_others = statistics.mean(other_avgs)
        difference = first_avg - avg_others
        
        if difference > threshold:
            return BiasWarning(
                bias_type="anchoring",
                severity="low",
                message=f"ℹ️ First option '{first_option_name}' might be anchoring your judgment",
                evidence=f"First option avg: {first_avg:.1f}, Others avg: {avg_others:.1f} (Δ{difference:.1f})",
                suggestion="First options often get inflated scores (anchoring bias). Double-check if this is genuinely better."
            )
        
        return None
    
    def get_stats(self) -> Dict:
        """Get bias detector statistics."""
        return {
            'warnings_generated': self.warnings_generated
        }
