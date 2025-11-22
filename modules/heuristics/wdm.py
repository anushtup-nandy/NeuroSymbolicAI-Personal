"""
Weighted Decision Matrix (WDM)
===============================
Classic multi-criteria decision model with normalized weights.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class WeightedDecisionMatrix:
    """
    Weighted Decision Matrix (WDM).
    
    Formula: Score = Σ(weight_i × score_i)
    
    Optimistic model that maximizes weighted sum of scores.
    Good for stable preferences with clear factor importance.
    """
    
    @staticmethod
    def get_name() -> str:
        return "Weighted Decision Matrix"
    
    @staticmethod
    def get_description() -> str:
        return "Optimistic model: Maximizes weighted sum of scores"
    
    def evaluate(self, options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """
        Evaluate options using WDM.
        
        Args:
            options: List of {'name': str, 'scores': {factor_name: score}}
            factors: List of {'name': str, 'weight': float}
            
        Returns:
            DataFrame sorted by WDM_Score (descending)
        """
        factor_names = [f['name'] for f in factors]
        weights = np.array([f['weight'] for f in factors])
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        results = []
        for opt in options:
            scores_vec = np.array([opt['scores'].get(f, 0.0) for f in factor_names])
            weighted_score = np.dot(scores_vec, weights)
            
            results.append({
                "Option": opt['name'],
                "WDM_Score": round(weighted_score, 3),
                **{f: opt['scores'].get(f, 0) for f in factor_names}
            })
        
        return pd.DataFrame(results).sort_values(by="WDM_Score", ascending=False)
