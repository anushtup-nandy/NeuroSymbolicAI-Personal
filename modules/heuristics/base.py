"""
Decision Heuristics Base
=========================
Abstract interface for decision-making models.
"""

from typing import List, Dict, Protocol
import pandas as pd


class DecisionHeuristic(Protocol):
    """
    Protocol (interface) for decision heuristics.
    
    All decision models must implement these methods.
    """
    
    def evaluate(self, options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """
        Evaluate options and return ranked results.
        
        Args:
            options: List of option dicts with 'name' and 'scores'
            factors: List of factor dicts with 'name' and 'weight'
            
        Returns:
            DataFrame with ranked results
        """
        ...
    
    @staticmethod
    def get_name() -> str:
        """Return human-readable model name."""
        ...
    
    @staticmethod
    def get_description() -> str:
        """Return model description and use case."""
        ...


class DecisionModels:
    """
    Container for all decision models (for backward compatibility).
    """
    
    @staticmethod
    def weighted_decision_matrix(options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """Alias for WDM model."""
        from modules.heuristics.wdm import WeightedDecisionMatrix
        return WeightedDecisionMatrix().evaluate(options, factors)
    
    @staticmethod
    def minimax_regret(options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """Alias for Minimax Regret model."""
        from modules.heuristics.minimax_regret import MinimaxRegret
        return MinimaxRegret().evaluate(options, factors)
    
    @staticmethod
    def topsis(options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """Alias for TOPSIS model."""
        from modules.heuristics.topsis import TOPSIS
        return TOPSIS().evaluate(options, factors)
    
    @staticmethod
    def bayesian_decision(options: List[Dict], factors: List[Dict], 
                          prior_probs: Dict = None) -> pd.DataFrame:
        """Alias for Bayesian model."""
        from modules.heuristics.bayesian import BayesianDecision
        return BayesianDecision().evaluate(options, factors, prior_probs)
