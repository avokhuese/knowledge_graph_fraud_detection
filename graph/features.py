import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

class GraphFeatureExtractor:
    """Extracts graph-structural features from the provider graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.provider_nodes = self._get_provider_nodes()
        
    def _get_provider_nodes(self) -> List[str]:
        """Get all provider nodes"""
        return [
            node for node, attr in self.graph.nodes(data=True)
            if attr.get('node_type') in ['Provider', 'Provider_Ind', 'Provider_Org']
        ]
    
    def compute_centrality_features(self) -> pd.DataFrame:
        """Compute various centrality measures"""
        providers = self.provider_nodes
        
        # Extract subgraphs for each edge type
        edge_types = set(
            attr.get('edge_type', 'Unknown') 
            for _, _, attr in self.graph.edges(data=True)
        )
        
        features = pd.DataFrame(index=providers)
        
        for edge_type in edge_types:
            # Create subgraph with only this edge type
            subgraph = nx.DiGraph()
            for u, v, attr in self.graph.edges(data=True):
                if attr.get('edge_type') == edge_type:
                    subgraph.add_edge(u, v)
            
            if subgraph.number_of_edges() == 0:
                continue
            
            # Degree centrality
            try:
                degree_cent = nx.degree_centrality(subgraph)
                features[f'degree_cent_{edge_type}'] = pd.Series(degree_cent)
            except:
                pass
            
            # Betweenness centrality (expensive, sample if needed)
            if subgraph.number_of_nodes() < 10000:
                try:
                    betweenness = nx.betweenness_centrality(subgraph, k=min(1000, subgraph.number_of_nodes()))
                    features[f'betweenness_cent_{edge_type}'] = pd.Series(betweenness)
                except:
                    pass
            
            # Eigenvector centrality
            try:
                eigenvector = nx.eigenvector_centrality_numpy(subgraph)
                features[f'eigenvector_cent_{edge_type}'] = pd.Series(eigenvector)
            except:
                pass
        
        return features.fillna(0)
    
    def compute_pagerank_from_seeds(self, seed_providers: List[str], alpha: float = 0.85) -> pd.Series:
        """
        Personalized PageRank seeded from known excluded providers
        This is the guilt-by-association baseline
        """
        # Create undirected version for PageRank
        G_undirected = nx.Graph()
        for u, v, attr in self.graph.edges(data=True):
            G_undirected.add_edge(u, v, weight=1.0)
        
        # Create personalization vector
        personalization = {}
        for node in G_undirected.nodes():
            if node in seed_providers:
                personalization[node] = 1.0
            else:
                personalization[node] = 0.0
        
        # Normalize
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        
        # Compute PageRank
        pagerank = nx.pagerank(
            G_undirected, 
            alpha=alpha, 
            personalization=personalization,
            max_iter=100
        )
        
        return pd.Series(pagerank, name='pagerank_risk_score')
    
    def compute_community_features(self) -> pd.DataFrame:
        """Community detection and related features"""
        # Use Louvain community detection
        try:
            from community import community_louvain
            
            # Convert to undirected
            G_undirected = nx.Graph()
            for u, v in self.graph.edges():
                G_undirected.add_edge(u, v)
            
            # Detect communities
            partition = community_louvain.best_partition(G_undirected)
            
            # Compute community statistics
            community_stats = defaultdict(lambda: {
                'size': 0, 
                'fraud_count': 0, 
                'providers': []
            })
            
            for node, community_id in partition.items():
                community_stats[community_id]['size'] += 1
                community_stats[community_id]['providers'].append(node)
                
                # Count known fraud providers
                if self.graph.nodes[node].get('is_excluded', False):
                    community_stats[community_id]['fraud_count'] += 1
            
            # Create features
            features = pd.DataFrame(index=self.provider_nodes)
            community_ids = []
            community_sizes = []
            community_fraud_rates = []
            
            for provider in self.provider_nodes:
                if provider in partition:
                    comm_id = partition[provider]
                    community_ids.append(comm_id)
                    community_sizes.append(community_stats[comm_id]['size'])
                    fraud_rate = (
                        community_stats[comm_id]['fraud_count'] / 
                        community_stats[comm_id]['size']
                    )
                    community_fraud_rates.append(fraud_rate)
                else:
                    community_ids.append(-1)
                    community_sizes.append(0)
                    community_fraud_rates.append(0)
            
            features['community_id'] = community_ids
            features['community_size'] = community_sizes
            features['community_fraud_rate'] = community_fraud_rates
            
            return features
            
        except ImportError:
            print("python-louvain not installed, skipping community detection")
            return pd.DataFrame(index=self.provider_nodes)
    
    def compute_local_density_features(self) -> pd.DataFrame:
        """Clustering coefficient and ego-network features"""
        features = pd.DataFrame(index=self.provider_nodes)
        
        # Clustering coefficient
        G_undirected = nx.Graph()
        for u, v in self.graph.edges():
            if u in self.provider_nodes and v in self.provider_nodes:
                G_undirected.add_edge(u, v)
        
        clustering_coeffs = nx.clustering(G_undirected)
        features['clustering_coefficient'] = pd.Series(clustering_coeffs)
        
        # Ego network size
        ego_sizes = {}
        for provider in self.provider_nodes:
            ego = nx.ego_graph(G_undirected, provider, radius=1)
            ego_sizes[provider] = ego.number_of_nodes()
        
        features['ego_network_size'] = pd.Series(ego_sizes)
        
        return features.fillna(0)
    
    def compute_distance_to_bad_actors(self, seed_providers: List[str]) -> pd.Series:
        """Shortest path distance to known excluded providers"""
        distances = {}
        
        G_undirected = nx.Graph()
        for u, v in self.graph.edges():
            G_undirected.add_edge(u, v)
        
        for provider in self.provider_nodes:
            min_distance = float('inf')
            
            for seed in seed_providers:
                try:
                    distance = nx.shortest_path_length(G_undirected, provider, seed)
                    min_distance = min(min_distance, distance)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            distances[provider] = min_distance if min_distance != float('inf') else -1
        
        return pd.Series(distances, name='min_distance_to_bad_actor')
    
    def extract_all_structural_features(self, seed_providers: List[str]) -> pd.DataFrame:
        """Extract all graph-structural features"""
        print("Computing centrality features...")
        centrality_features = self.compute_centrality_features()
        
        print("Computing PageRank from seeds...")
        pagerank_scores = self.compute_pagerank_from_seeds(seed_providers)
        
        print("Computing community features...")
        community_features = self.compute_community_features()
        
        print("Computing local density features...")
        density_features = self.compute_local_density_features()
        
        print("Computing distance to bad actors...")
        distance_features = self.compute_distance_to_bad_actors(seed_providers)
        
        # Combine all features
        all_features = pd.concat([
            centrality_features,
            pagerank_scores,
            community_features,
            density_features,
            distance_features
        ], axis=1)
        
        return all_features