import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import Dict, Tuple, List

class FraudDetectionEvaluator:
    """Evaluation metrics for fraud detection"""
    
    def __init__(self, investigator_capacity: int = 50):
        self.investigator_capacity = investigator_capacity
        
    def precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int = None) -> float:
        """Precision@K - what fraction of top K are actually fraud"""
        if k is None:
            k = self.investigator_capacity
        
        # Get top K indices
        top_k_idx = np.argsort(y_scores)[-k:]
        top_k_true = y_true[top_k_idx]
        
        return top_k_true.sum() / k
    
    def recall_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int = None) -> float:
        """Recall@K - what fraction of all fraud cases are in top K"""
        if k is None:
            k = self.investigator_capacity
        
        total_fraud = y_true.sum()
        if total_fraud == 0:
            return 0.0
        
        # Get top K indices
        top_k_idx = np.argsort(y_scores)[-k:]
        top_k_true = y_true[top_k_idx]
        
        return top_k_true.sum() / total_fraud
    
    def calculate_lift(self, 
                      baseline_scores: np.ndarray,
                      model_scores: np.ndarray,
                      y_true: np.ndarray,
                      k: int = None) -> float:
        """Calculate lift over baseline system"""
        if k is None:
            k = self.investigator_capacity
        
        baseline_precision = self.precision_at_k(y_true, baseline_scores, k)
        model_precision = self.precision_at_k(y_true, model_scores, k)
        
        if baseline_precision == 0:
            return float('inf') if model_precision > 0 else 0.0
        
        return (model_precision - baseline_precision) / baseline_precision
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_scores: np.ndarray,
                baseline_scores: np.ndarray = None) -> Dict:
        """Comprehensive evaluation"""
        
        results = {
            'auc_pr': average_precision_score(y_true, y_scores),
            'auc_roc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
            'precision_at_k': self.precision_at_k(y_true, y_scores),
            'recall_at_k': self.recall_at_k(y_true, y_scores),
        }
        
        # Add precision-recall curve data
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        results['pr_curve'] = {
            'precision': precisions,
            'recall': recalls,
            'thresholds': thresholds
        }
        
        # Calculate lift if baseline provided
        if baseline_scores is not None:
            results['lift_at_k'] = self.calculate_lift(
                baseline_scores, y_scores, y_true
            )
        
        # Calculate optimal threshold based on F1
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        results['optimal_threshold'] = thresholds[optimal_idx]
        results['optimal_f1'] = f1_scores[optimal_idx]
        
        return results
    
    def generate_report(self, 
                       provider_ids: List[str],
                       y_true: np.ndarray,
                       y_scores: np.ndarray,
                       top_k: int = 20) -> pd.DataFrame:
        """Generate report of top-risk providers"""
        
        # Create dataframe
        df = pd.DataFrame({
            'provider_id': provider_ids,
            'risk_score': y_scores,
            'is_fraud': y_true
        })
        
        # Sort by risk score
        df = df.sort_values('risk_score', ascending=False)
        
        # Add rank
        df['rank'] = range(1, len(df) + 1)
        
        # Add cumulative metrics
        df['cumulative_fraud'] = df['is_fraud'].cumsum()
        df['cumulative_precision'] = df['cumulative_fraud'] / df['rank']
        
        return df.head(top_k)