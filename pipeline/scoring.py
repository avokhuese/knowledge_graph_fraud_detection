import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

from ..config.settings import Settings
from ..data.ingestion import DataIngestion
from ..graph.construction import GraphBuilder
from ..graph.features import GraphFeatureExtractor
from ..models.gnn import FraudGNN, GNNTrainer
from ..models.hybrid import HybridFraudDetector
from ..evaluation.metrics import FraudDetectionEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """Main pipeline orchestrating the fraud detection workflow"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ingestion = DataIngestion(settings)
        self.graph_builder = GraphBuilder(settings)
        self.evaluator = FraudDetectionEvaluator(settings.pipeline.investigator_capacity)
        
    def run(self, 
            provider_file: str,
            claims_file: str,
            exclusions_file: str,
            ownership_file: Optional[str] = None,
            bank_accounts_file: Optional[str] = None) -> Dict:
        """
        Execute the full fraud detection pipeline
        
        Returns:
            Dictionary containing risk scores, explanations, and evaluation metrics
        """
        
        logger.info("Starting fraud detection pipeline...")
        
        # 1. Data Ingestion
        logger.info("Step 1: Ingesting data...")
        providers = self.ingestion.load_providers(provider_file)
        claims = self.ingestion.load_claims(claims_file)
        exclusions = self.ingestion.load_exclusions(exclusions_file)
        
        ownership = None
        if ownership_file:
            ownership = self.ingestion.load_ownership(ownership_file)
        
        bank_accounts = None
        if bank_accounts_file:
            bank_accounts = pd.read_csv(bank_accounts_file)
        
        logger.info(f"Loaded {len(providers)} providers, {len(claims)} claims, "
                   f"{len(exclusions)} exclusion records")
        
        # 2. Graph Construction
        logger.info("Step 2: Building graph...")
        graph = self.graph_builder.build_graph(
            providers, claims, exclusions, ownership, bank_accounts
        )
        
        stats = self.graph_builder.get_graph_statistics()
        logger.info(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # 3. Feature Engineering
        logger.info("Step 3: Extracting features...")
        
        # Get seed providers (known exclusions)
        seed_providers = exclusions['entity_npi'].unique().tolist()
        
        # Extract graph-structural features
        feature_extractor = GraphFeatureExtractor(graph)
        graph_features = feature_extractor.extract_all_structural_features(seed_providers)
        
        # Compute PageRank baseline scores
        baseline_scores = feature_extractor.compute_pagerank_from_seeds(seed_providers)
        
        logger.info(f"Extracted {graph_features.shape[1]} graph features")
        
        # 4. GNN Training and Embeddings
        logger.info("Step 4: Training GNN model...")
        
        # Convert to PyTorch Geometric format (simplified for illustration)
        # In practice, this would be a more complex conversion
        gnn_embeddings = self._train_gnn_and_extract_embeddings(graph, providers)
        
        # 5. Hybrid Model Training
        logger.info("Step 5: Training hybrid model...")
        
        # Prepare labels (from exclusions)
        labels = pd.Series(0, index=providers['npi'])
        labels[providers['npi'].isin(seed_providers)] = 1
        
        # Prepare features
        hybrid_detector = HybridFraudDetector(random_state=self.settings.pipeline.random_seed)
        X = hybrid_detector.prepare_features(
            graph_features,
            self._compute_claims_features(claims, providers),
            gnn_embeddings,
            providers['npi']
        )
        
        # Train hybrid model
        results = hybrid_detector.train(X, labels)
        
        # 6. Generate Risk Scores
        logger.info("Step 6: Generating risk scores...")
        risk_scores = hybrid_detector.predict_risk_scores(X)
        
        # 7. Generate Explanations
        logger.info("Step 7: Generating explanations...")
        explanations = hybrid_detector.explain_predictions(X)
        
        # 8. Evaluation
        logger.info("Step 8: Evaluating model...")
        evaluation = self.evaluator.evaluate(
            labels.values,
            risk_scores,
            baseline_scores.values if baseline_scores is not None else None
        )
        
        # 9. Compile Results
        results = {
            'providers': providers['npi'].tolist(),
            'risk_scores': risk_scores.tolist(),
            'baseline_scores': baseline_scores.tolist() if baseline_scores is not None else None,
            'labels': labels.tolist(),
            'evaluation': evaluation,
            'explanations': {
                'feature_importance': explanations['feature_importance'],
                'top_features': explanations['top_features']
            },
            'graph_statistics': stats,
            'metadata': {
                'pipeline_version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'num_providers': len(providers),
                'num_claims': len(claims),
                'num_exclusions': len(exclusions)
            }
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("Pipeline completed successfully!")
        return results
    
    def _train_gnn_and_extract_embeddings(self, 
                                          graph: nx.MultiDiGraph, 
                                          providers: pd.DataFrame) -> np.ndarray:
        """Train GNN and extract embeddings (simplified)"""
        # This is a simplified placeholder
        # In practice, you would convert the NetworkX graph to PyG format
        # and train the GNN properly
        
        # For now, return random embeddings as placeholder
        np.random.seed(self.settings.pipeline.random_seed)
        return np.random.randn(len(providers), self.settings.model.embedding_dim)
    
    def _compute_claims_features(self, 
                                 claims: pd.DataFrame, 
                                 providers: pd.DataFrame) -> pd.DataFrame:
        """Compute claims-behavioral features"""
        features = pd.DataFrame(index=providers['npi'])
        
        # Billing volume
        claim_counts = claims.groupby('provider_npi').size()
        features['claim_count'] = claim_counts
        
        # Total billed amount
        total_billed = claims.groupby('provider_npi')['billed_amount'].sum()
        features['total_billed'] = total_billed
        
        # Average billed per claim
        features['avg_billed'] = total_billed / claim_counts
        
        # Patient count
        if 'beneficiary_hash' in claims.columns:
            patient_counts = claims.groupby('provider_npi')['beneficiary_hash'].nunique()
            features['unique_patients'] = patient_counts
        
        # Temporal features
        if 'date_of_service' in claims.columns:
            claims['date_of_service'] = pd.to_datetime(claims['date_of_service'])
            date_ranges = claims.groupby('provider_npi')['date_of_service'].agg(['min', 'max'])
            features['active_days'] = (date_ranges['max'] - date_ranges['min']).dt.days
            features['claims_per_day'] = features['claim_count'] / features['active_days'].clip(lower=1)
        
        return features.fillna(0)
    
    def _save_results(self, results: Dict):
        """Save results to disk"""
        output_dir = self.settings.pipeline.output_dir
        
        # Save risk scores
        scores_df = pd.DataFrame({
            'provider_id': results['providers'],
            'risk_score': results['risk_scores'],
            'baseline_score': results.get('baseline_scores'),
            'label': results['labels']
        })
        scores_df.to_csv(f"{output_dir}/risk_scores.csv", index=False)
        
        # Save evaluation metrics
        with open(f"{output_dir}/evaluation.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            eval_copy = results['evaluation'].copy()
            if 'pr_curve' in eval_copy:
                eval_copy['pr_curve'] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in eval_copy['pr_curve'].items()
                }
            json.dump(eval_copy, f, indent=2)
        
        # Save explanations
        with open(f"{output_dir}/explanations.json", 'w') as f:
            json.dump(results['explanations'], f, indent=2)
        
        # Save metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")