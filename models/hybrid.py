import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, precision_recall_curve
import shap
from typing import Dict, Tuple, Optional
import pickle

class HybridFraudDetector:
    """
    Hybrid model combining GNN embeddings with gradient-boosted trees
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_features(self,
                        graph_features: pd.DataFrame,
                        claims_features: pd.DataFrame,
                        gnn_embeddings: np.ndarray,
                        provider_index: pd.Index) -> pd.DataFrame:
        """Combine all feature sources"""
        
        # Align all features by provider ID
        combined = pd.DataFrame(index=provider_index)
        
        if graph_features is not None:
            combined = combined.join(graph_features, how='left')
        
        if claims_features is not None:
            combined = combined.join(claims_features, how='left')
        
        # Add GNN embeddings
        if gnn_embeddings is not None:
            embedding_df = pd.DataFrame(
                gnn_embeddings,
                index=provider_index,
                columns=[f'gnn_emb_{i}' for i in range(gnn_embeddings.shape[1])]
            )
            combined = combined.join(embedding_df, how='left')
        
        self.feature_names = combined.columns.tolist()
        
        return combined.fillna(0)
    
    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              validation_size: float = 0.2) -> Dict:
        """Train XGBoost classifier"""
        
        # Split data temporally if possible
        split_idx = int(len(X) * (1 - validation_size))
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        # Handle class imbalance with scale_pos_weight
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': ['aucpr', 'logloss'],
            'early_stopping_rounds': 20
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=10
        )
        
        # Training history
        results = self.model.evals_result()
        
        return results
    
    def predict_risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Generate fraud risk scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def explain_predictions(self, X: pd.DataFrame, top_k: int = 10) -> Dict:
        """Generate SHAP explanations for predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        explanations = {
            'shap_values': shap_values,
            'feature_importance': dict(zip(
                self.feature_names,
                np.abs(shap_values).mean(0)
            )),
            'top_features': self.feature_names[
                np.argsort(np.abs(shap_values).mean(0))[-top_k:]
            ].tolist()
        }
        
        return explanations
    
    def save_model(self, filepath: str):
        """Save model and metadata"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HybridFraudDetector':
        """Load saved model"""
        detector = cls()
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        detector.model = model_data['model']
        detector.feature_names = model_data['feature_names']
        return detector