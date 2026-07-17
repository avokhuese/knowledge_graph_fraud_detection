"""
Baseline models for fraud detection
Includes PageRank-based guilt-by-association and traditional ML baselines
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

class PageRankBaseline:
    """
    PageRank-based guilt-by-association baseline
    Propagates risk from known excluded providers through the graph
    """
    
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize PageRank baseline
        
        Args:
            alpha: Damping factor (probability of continuing random walk)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.pagerank_scores = None
        self.graph = None
        
    def fit(self, graph: nx.Graph, seed_nodes: List[str]) -> pd.Series:
        """
        Compute personalized PageRank from seed nodes
        
        Args:
            graph: NetworkX graph (can be directed or undirected)
            seed_nodes: List of known fraud/exclusion provider IDs
            
        Returns:
            Series with PageRank scores for all nodes
        """
        self.graph = graph
        
        # Convert to undirected for PageRank if needed
        if isinstance(graph, nx.DiGraph) or isinstance(graph, nx.MultiDiGraph):
            G = nx.Graph()
            for u, v, data in graph.edges(data=True):
                if G.has_edge(u, v):
                    G[u][v]['weight'] = G[u][v].get('weight', 0) + 1
                else:
                    G.add_edge(u, v, weight=1)
        else:
            G = graph.copy()
        
        # Create personalization vector
        personalization = {}
        for node in G.nodes():
            if node in seed_nodes:
                personalization[node] = 1.0
            else:
                personalization[node] = 0.0
        
        # Normalize personalization vector
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        
        # Compute personalized PageRank
        try:
            self.pagerank_scores = nx.pagerank(
                G,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol
            )
        except Exception as e:
            print(f"PageRank computation failed: {e}")
            # Fallback to uniform scores
            self.pagerank_scores = {node: 1.0/len(G) for node in G.nodes()}
        
        return pd.Series(self.pagerank_scores, name='pagerank_risk_score')
    
    def predict_risk(self, node_ids: List[str]) -> np.ndarray:
        """
        Get risk scores for specific nodes
        
        Args:
            node_ids: List of node IDs to get scores for
            
        Returns:
            Array of risk scores
        """
        if self.pagerank_scores is None:
            raise ValueError("Model not fitted yet")
        
        scores = []
        for node_id in node_ids:
            scores.append(self.pagerank_scores.get(node_id, 0.0))
        
        return np.array(scores)
    
    def get_top_risk_providers(self, n: int = 20, exclude_seeds: bool = True,
                               seed_nodes: List[str] = None) -> pd.DataFrame:
        """
        Get top N highest risk providers
        
        Args:
            n: Number of top providers to return
            exclude_seeds: Whether to exclude seed nodes
            seed_nodes: List of seed nodes to exclude
            
        Returns:
            DataFrame with top risk providers
        """
        if self.pagerank_scores is None:
            raise ValueError("Model not fitted yet")
        
        # Convert to DataFrame
        scores_df = pd.DataFrame([
            {'provider': node, 'risk_score': score}
            for node, score in self.pagerank_scores.items()
        ])
        
        # Exclude seed nodes if requested
        if exclude_seeds and seed_nodes:
            scores_df = scores_df[~scores_df['provider'].isin(seed_nodes)]
        
        # Sort and get top N
        scores_df = scores_df.sort_values('risk_score', ascending=False)
        return scores_df.head(n)
    
    def evaluate(self, y_true: pd.Series, provider_ids: List[str] = None) -> Dict:
        """
        Evaluate PageRank baseline performance
        
        Args:
            y_true: True labels (1 for fraud, 0 for non-fraud)
            provider_ids: Optional list of provider IDs to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.pagerank_scores is None:
            raise ValueError("Model not fitted yet")
        
        if provider_ids is None:
            provider_ids = list(self.pagerank_scores.keys())
        
        # Get scores for providers with labels
        scores = []
        labels = []
        
        for pid in provider_ids:
            if pid in y_true.index and pid in self.pagerank_scores:
                scores.append(self.pagerank_scores[pid])
                labels.append(y_true[pid])
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        if len(labels) == 0:
            return {'error': 'No matching providers found'}
        
        # Calculate metrics
        metrics = {
            'auc_pr': average_precision_score(labels, scores),
            'auc_roc': roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5,
            'mean_score_fraud': scores[labels == 1].mean() if labels.sum() > 0 else 0,
            'mean_score_nonfraud': scores[labels == 0].mean() if (labels == 0).sum() > 0 else 0,
            'num_evaluated': len(scores)
        }
        
        return metrics


class RandomWalkBaseline:
    """
    Random Walk with Restart (RWR) from seed nodes
    Similar to PageRank but with different normalization
    """
    
    def __init__(self, restart_prob: float = 0.15, max_iter: int = 100):
        """
        Initialize RWR baseline
        
        Args:
            restart_prob: Probability of restarting from seed nodes
            max_iter: Maximum iterations
        """
        self.restart_prob = restart_prob
        self.max_iter = max_iter
        self.scores = None
        
    def fit(self, graph: nx.Graph, seed_nodes: List[str]) -> pd.Series:
        """
        Compute RWR scores
        
        Args:
            graph: NetworkX graph
            seed_nodes: Seed fraud provider IDs
            
        Returns:
            Series with RWR scores
        """
        # Get adjacency matrix
        nodes = list(graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build transition matrix
        A = nx.adjacency_matrix(graph)
        # Add self-loops to avoid dangling nodes
        A = A + csr_matrix(np.eye(n))
        # Normalize
        row_sums = np.array(A.sum(axis=1)).flatten()
        D_inv = csr_matrix(np.diag(1.0 / np.maximum(row_sums, 1e-10)))
        P = D_inv @ A
        
        # Create restart vector
        restart = np.zeros(n)
        seed_indices = [node_to_idx[node] for node in seed_nodes if node in node_to_idx]
        if len(seed_indices) > 0:
            restart[seed_indices] = 1.0 / len(seed_indices)
        
        # Power iteration
        v = restart.copy()
        for _ in range(self.max_iter):
            v_new = (1 - self.restart_prob) * P.T @ v + self.restart_prob * restart
            if np.linalg.norm(v_new - v) < 1e-6:
                break
            v = v_new
        
        self.scores = pd.Series(v, index=nodes, name='rwr_risk_score')
        return self.scores
    
    def predict_risk(self, node_ids: List[str]) -> np.ndarray:
        """Get risk scores for specific nodes"""
        if self.scores is None:
            raise ValueError("Model not fitted yet")
        return np.array([self.scores.get(nid, 0.0) for nid in node_ids])


class TraditionalMLBaseline:
    """
    Traditional ML baselines without graph features
    Uses only claims-behavioral features for comparison
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize traditional ML baseline
        
        Args:
            model_type: Type of model ('random_forest', 'logistic', 'isolation_forest')
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=random_state
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=random_state
            )
        elif model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TraditionalMLBaseline':
        """
        Train the baseline model
        
        Args:
            X: Feature matrix (claims-behavioral features only)
            y: Labels (not needed for unsupervised methods)
            
        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # Train model
        if self.model_type == 'isolation_forest':
            self.model.fit(X_scaled)
        else:
            if y is None:
                raise ValueError("Labels required for supervised models")
            self.model.fit(X_scaled, y)
        
        return self
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate risk scores
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of risk scores
        """
        X_scaled = self.scaler.transform(X.fillna(0))
        
        if self.model_type == 'isolation_forest':
            # Convert anomaly scores to risk scores (higher = more anomalous)
            scores = -self.model.score_samples(X_scaled)
            # Normalize to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        else:
            scores = self.model.predict_proba(X_scaled)[:, 1]
        
        return scores
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        elif self.model_type == 'logistic':
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()
        
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return imp_df


class LabelPropagationBaseline:
    """
    Label propagation from known fraud cases
    Spreads labels through the graph structure
    """
    
    def __init__(self, n_iterations: int = 20, alpha: float = 0.8):
        """
        Initialize label propagation
        
        Args:
            n_iterations: Number of propagation iterations
            alpha: Clamping factor (how much to retain original labels)
        """
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.label_scores = None
        
    def fit(self, graph: nx.Graph, seed_nodes: List[str]) -> pd.Series:
        """
        Propagate labels from seed nodes
        
        Args:
            graph: NetworkX graph
            seed_nodes: Known fraud provider IDs
            
        Returns:
            Series with propagated risk scores
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Initialize labels
        labels = np.zeros(n)
        seed_indices = [node_to_idx[node] for node in seed_nodes if node in node_to_idx]
        labels[seed_indices] = 1.0
        
        # Create transition matrix
        A = nx.adjacency_matrix(graph)
        # Normalize
        row_sums = np.array(A.sum(axis=1)).flatten()
        D_inv = np.diag(1.0 / np.maximum(row_sums, 1e-10))
        T = D_inv @ A.toarray()
        
        # Label propagation
        Y = labels.copy()
        for _ in range(self.n_iterations):
            Y_new = self.alpha * T @ Y + (1 - self.alpha) * labels
            Y = Y_new
        
        self.label_scores = pd.Series(Y, index=nodes, name='label_propagation_score')
        return self.label_scores
    
    def predict_risk(self, node_ids: List[str]) -> np.ndarray:
        """Get propagated risk scores"""
        if self.label_scores is None:
            raise ValueError("Model not fitted yet")
        return np.array([self.label_scores.get(nid, 0.0) for nid in node_ids])


class BaselineEnsemble:
    """
    Ensemble of baseline methods for robust risk scoring
    """
    
    def __init__(self):
        self.pagerank = PageRankBaseline()
        self.rwr = RandomWalkBaseline()
        self.label_prop = LabelPropagationBaseline()
        self.scores = {}
        
    def fit(self, graph: nx.Graph, seed_nodes: List[str]) -> pd.DataFrame:
        """
        Run all baseline methods
        
        Args:
            graph: NetworkX graph
            seed_nodes: Known fraud provider IDs
            
        Returns:
            DataFrame with all baseline scores
        """
        print("Computing PageRank baseline...")
        pr_scores = self.pagerank.fit(graph, seed_nodes)
        self.scores['pagerank'] = pr_scores
        
        try:
            print("Computing Random Walk with Restart...")
            rwr_scores = self.rwr.fit(graph, seed_nodes)
            self.scores['rwr'] = rwr_scores
        except Exception as e:
            print(f"RWR failed: {e}")
            self.scores['rwr'] = pd.Series(0, index=pr_scores.index)
        
        try:
            print("Computing Label Propagation...")
            lp_scores = self.label_prop.fit(graph, seed_nodes)
            self.scores['label_propagation'] = lp_scores
        except Exception as e:
            print(f"Label propagation failed: {e}")
            self.scores['label_propagation'] = pd.Series(0, index=pr_scores.index)
        
        # Create ensemble DataFrame
        ensemble_df = pd.DataFrame(self.scores)
        
        # Add ensemble score (average)
        ensemble_df['ensemble_score'] = ensemble_df.mean(axis=1)
        
        return ensemble_df
    
    def get_consensus_risk(self, n_top: int = 20) -> pd.DataFrame:
        """
        Get providers that are high-risk across multiple baselines
        
        Args:
            n_top: Number of top providers to return
            
        Returns:
            DataFrame with top consensus high-risk providers
        """
        if not self.scores:
            raise ValueError("Models not fitted yet")
        
        df = pd.DataFrame(self.scores)
        
        # Rank providers in each method
        for col in df.columns:
            df[f'{col}_rank'] = df[col].rank(ascending=False)
        
        # Calculate average rank
        rank_cols = [c for c in df.columns if c.endswith('_rank')]
        df['avg_rank'] = df[rank_cols].mean(axis=1)
        
        # Get top consensus providers
        top_consensus = df.nsmallest(n_top, 'avg_rank')
        
        return top_consensus