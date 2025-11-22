"""
TOPSIS
======
Technique for Order Preference by Similarity to Ideal Solution.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class TOPSIS:
    """
    TOPSIS: Technique for Order Preference by Similarity to Ideal Solution.
    
    Finds option closest to ideal solution and farthest from negative-ideal solution.
    Uses Euclidean distance in normalized, weighted space.
    
    Good for finding balanced, compromise solutions.
    """
    
    @staticmethod
    def get_name() -> str:
        return "TOPSIS"
    
    @staticmethod
    def get_description() -> str:
        return "Distance-based model: Finds option closest to ideal solution"
    
    def evaluate(self, options: List[Dict], factors: List[Dict]) -> pd.DataFrame:
        """
        Evaluate options using TOPSIS.
        
        Args:
            options: List of {'name': str, 'scores': {factor_name: score}}
            factors: List of {'name': str, 'weight': float}
            
        Returns:
            DataFrame sorted by TOPSIS_Score (descending)
        """
        factor_names = [f['name'] for f in factors]
        weights = np.array([f['weight'] for f in factors])
        weights = weights / weights.sum()
        
        # Build decision matrix
        matrix = []
        for opt in options:
            row = [opt['scores'].get(f, 0) for f in factor_names]
            matrix.append(row)
        matrix = np.array(matrix)
        
        # Normalize matrix (vector normalization)
        norms = np.sqrt(np.sum(matrix**2, axis=0))
        norms[norms == 0] = 1  # Avoid division by zero
        norm_matrix = matrix / norms
        
        # Weight normalized matrix
        weighted_matrix = norm_matrix * weights
        
        # Ideal and negative-ideal solutions
        ideal = np.max(weighted_matrix, axis=0)
        negative_ideal = np.min(weighted_matrix, axis=0)
        
        # Calculate distances
        results = []
        for i, opt in enumerate(options):
            dist_ideal = np.sqrt(np.sum((weighted_matrix[i] - ideal)**2))
            dist_neg_ideal = np.sqrt(np.sum((weighted_matrix[i] - negative_ideal)**2))
            
            # TOPSIS score: closeness to ideal
            score = dist_neg_ideal / (dist_ideal + dist_neg_ideal + 1e-10)
            
            results.append({
                "Option": opt['name'],
                "TOPSIS_Score": round(score, 3),
                "Dist_to_Ideal": round(dist_ideal, 3),
                "Dist_to_Worst": round(dist_neg_ideal, 3)
            })
        
        return pd.DataFrame(results).sort_values(by="TOPSIS_Score", ascending=False)
