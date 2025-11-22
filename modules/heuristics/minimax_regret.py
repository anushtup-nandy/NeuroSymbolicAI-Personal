"""
Minimax Regret
==============
Risk-averse decision model that minimizes maximum opportunity cost.
"""

import pandas as pd
from typing import List, Dict


class MinimaxRegret:
    """
    Minimax Regret decision model.
    
    Formula: Regret(option, factor) = max_score(factor) - score(option, factor)
    
    Minimizes the maximum regret across all factors.
    Good for high-stakes, irreversible decisions.
    """
    
    @staticmethod
    def get_name() -> str:
        return "Minimax Regret"
    
    @staticmethod
    def get_description() -> str:
        return "Risk-averse model: Minimizes maximum opportunity cost"
    
    def evaluate(self, options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """
        Evaluate options using Minimax Regret.
        
        Args:
            options: List of {'name': str, 'scores': {factor_name: score}}
            factors: List of {'name': str, 'weight': float}
            
        Returns:
            DataFrame sorted by Max_Regret (ascending - lower is better)
        """
        factor_names = [f['name'] for f in factors]
        
        # Find max score per factor
        max_scores = {}
        for f in factor_names:
            max_scores[f] = max([opt['scores'].get(f, 0) for opt in options])
        
        results = []
        for opt in options:
            total_regret = 0
            regret_details = {}
            
            for f in factor_names:
                regret = max_scores[f] - opt['scores'].get(f, 0)
                weight = next(item['weight'] for item in factors if item['name'] == f)
                weighted_regret = regret * weight
                
                regret_details[f"{f}_Regret"] = round(weighted_regret, 3)
                total_regret += weighted_regret
            
            results.append({
                "Option": opt['name'],
                "Max_Regret": round(total_regret, 3),
                **regret_details
            })
        
        return pd.DataFrame(results).sort_values(by="Max_Regret", ascending=True)
