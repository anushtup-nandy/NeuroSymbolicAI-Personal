"""
Bayesian Decision Theory
=========================
Probabilistic decision model that updates beliefs based on evidence.
"""

import pandas as pd
from typing import List, Dict, Optional


class BayesianDecision:
    """
    Bayesian Decision Theory.
    
    Formula: P(Option | Evidence) ∝ P(Evidence | Option) × P(Option)
    
    Updates probability beliefs based on factor scores (evidence).
    Good for decisions where you want to update beliefs with new information.
    """
    
    @staticmethod
    def get_name() -> str:
        return "Bayesian Decision Theory"
    
    @staticmethod
    def get_description() -> str:
        return "Probabilistic model: Updates beliefs based on evidence"
    
    def evaluate(self, options: List[Dict], factors: List[Dict], 
                 prior_probs: Optional[Dict] = None) -> pd.DataFrame:
        """
        Evaluate options using Bayesian Decision Theory.
        
        Args:
            options: List of {'name': str, 'scores': {factor_name: score}}
            factors: List of {'name': str, 'weight': float}
            prior_probs: Optional dict of prior probabilities {option_name: prob}
            
        Returns:
            DataFrame sorted by Posterior_Normalized (descending)
        """
        if prior_probs is None:
            # Uniform prior
            prior_probs = {opt['name']: 1.0/len(options) for opt in options}
        
        factor_names = [f['name'] for f in factors]
        
        results = []
        for opt in options:
            # Likelihood: product of normalized scores (treating as probabilities)
            likelihood = 1.0
            for f in factor_names:
                score = opt['scores'].get(f, 0) / 10.0  # Normalize to [0, 1]
                likelihood *= (score + 0.1)  # Add smoothing
            
            # Posterior
            prior = prior_probs.get(opt['name'], 1.0/len(options))
            posterior = likelihood * prior
            
            results.append({
                "Option": opt['name'],
                "Prior": round(prior, 3),
                "Likelihood": round(likelihood, 3),
                "Posterior": round(posterior, 3)
            })
        
        df = pd.DataFrame(results)
        
        # Normalize posteriors
        total_posterior = df['Posterior'].sum()
        df['Posterior_Normalized'] = (df['Posterior'] / total_posterior).round(3)
        
        return df.sort_values(by="Posterior_Normalized", ascending=False)
