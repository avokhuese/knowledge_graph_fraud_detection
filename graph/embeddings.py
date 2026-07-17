"""
Graph embedding generation for fraud detection
Includes node2vec, metapath2vec, and graph autoencoder approaches
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

class Node2VecEmbeddings:
    """
    Node2Vec-style graph embeddings using random walks
    Simplified implementation without gensim dependency
    """
    
    def __init__(self, 
                 dimensions: int = 128, 
                 walk_length: int = 30,
                 num_walks: int = 200,
                 p: float = 1.0,
                 q: float = 1.0,
                 window_size: int = 10,
                 epochs: int = 5):
        """
        Initialize Node2Vec
        
        Args:
            dimensions: Embedding dimension
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            p: Return parameter (1/p probability of returning to source)
            q: In-out parameter (1/q probability of moving outward)
            window_size: Context window size
            epochs: Training epochs
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.epochs = epochs
        self.embeddings = None
        
    def _generate_walks(self, graph: nx.Graph) -> List[List[str]]:
        """
        Generate random walks using Node2Vec sampling strategy
        
        Args:
            graph: NetworkX graph
            
        Returns:
            List of random walks (each walk is list of node IDs)
        """
        walks = []
        nodes = list(graph.nodes())
        
        # Precompute transition probabilities
        alias_nodes = {}
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 0:
                # Uniform for first step
                probs = np.ones(len(neighbors)) / len(neighbors)
                alias_nodes[node] = self._alias_setup(probs)
        
        for _ in range(self.num_walks):
            # Shuffle nodes for each iteration
            np.random.shuffle(nodes)
            
            for node in nodes:
                walk = self._node2vec_walk(graph, node, alias_nodes)
                if len(walk) > 1:
                    walks.append(walk)
        
        return walks
    
    def _node2vec_walk(self, graph: nx.Graph, start_node: str, 
                       alias_nodes: Dict) -> List[str]:
        """
        Generate a single Node2Vec random walk
        
        Args:
            graph: NetworkX graph
            start_node: Starting node
            alias_nodes: Precomputed alias sampling tables
            
        Returns:
            List of nodes in the walk
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_neighbors = list(graph.neighbors(cur))
            
            if len(cur_neighbors) > 0:
                if len(walk) == 1:
                    # First step: no previous node
                    next_node = cur_neighbors[
                        self._alias_draw(
                            alias_nodes[cur][0], 
                            alias_nodes[cur][1]
                        )
                    ]
                else:
                    # Subsequent steps: consider previous node
                    prev = walk[-2]
                    next_node = self._get_next_node(graph, cur, prev)
                
                walk.append(next_node)
            else:
                break
        
        return walk
    
    def _get_next_node(self, graph: nx.Graph, current: str, previous: str) -> str:
        """
        Get next node in walk considering Node2Vec p,q parameters
        
        Args:
            graph: NetworkX graph
            current: Current node
            previous: Previous node
            
        Returns:
            Next node ID
        """
        neighbors = list(graph.neighbors(current))
        
        if len(neighbors) == 0:
            return current
        
        # Compute transition probabilities
        probs = []
        for neighbor in neighbors:
            if neighbor == previous:
                # Return to previous node
                probs.append(1.0 / self.p)
            elif graph.has_edge(neighbor, previous):
                # Distance 1 from previous
                probs.append(1.0)
            else:
                # Distance 2 from previous
                probs.append(1.0 / self.q)
        
        # Normalize
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # Sample next node
        next_node = np.random.choice(neighbors, p=probs)
        return next_node
    
    def _alias_setup(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set up alias sampling table
        
        Args:
            probs: Probability distribution
            
        Returns:
            Tuple of (alias_table, prob_table)
        """
        K = len(probs)
        q = probs * K
        J = np.zeros(K, dtype=int)
        smaller = []
        larger = []
        
        for i, prob in enumerate(q):
            if prob < 1.0:
                smaller.append(i)
            else:
                larger.append(i)
        
        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q
    
    def _alias_draw(self, J: np.ndarray, q: np.ndarray) -> int:
        """
        Draw sample from alias table
        
        Args:
            J: Alias table
            q: Probability table
            
        Returns:
            Sampled index
        """
        K = len(J)
        k = np.random.randint(0, K)
        
        if np.random.rand() < q[k]:
            return k
        else:
            return J[k]
    
    def fit(self, graph: nx.Graph) -> np.ndarray:
        """
        Generate Node2Vec embeddings
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Embedding matrix (n_nodes x dimensions)
        """
        # Convert to undirected
        if isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
            G = nx.Graph(graph)
        else:
            G = graph
        
        # Generate walks
        walks = self._generate_walks(G)
        
        # Simple skip-gram training using SVD
        # Build co-occurrence matrix
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        co_occurrence = np.zeros((n_nodes, n_nodes))
        
        for walk in walks:
            for i, node in enumerate(walk):
                node_idx = node_to_idx[node]
                # Context window
                start = max(0, i - self.window_size)
                end = min(len(walk), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_node = walk[j]
                        context_idx = node_to_idx[context_node]
                        # Weight by distance
                        weight = 1.0 / abs(i - j)
                        co_occurrence[node_idx, context_idx] += weight
        
        # Apply SVD for dimensionality reduction
        # Shift by small constant to ensure positive semi-definite
        co_occurrence = co_occurrence + 1e-10
        U, S, Vt = np.linalg.svd(co_occurrence, full_matrices=False)
        
        # Take top dimensions
        self.embeddings = U[:, :self.dimensions] @ np.diag(np.sqrt(S[:self.dimensions]))
        
        # Create DataFrame
        embedding_df = pd.DataFrame(
            self.embeddings,
            index=nodes,
            columns=[f'node2vec_{i}' for i in range(self.dimensions)]
        )
        
        return embedding_df
    
    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """
        Get embeddings for specific nodes
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            Embedding matrix for requested nodes
        """
        if self.embeddings is None:
            raise ValueError("Model not fitted yet")
        
        if isinstance(self.embeddings, pd.DataFrame):
            return self.embeddings.loc[node_ids].values
        
        return self.embeddings


class SpectralEmbeddings:
    """
    Spectral graph embeddings using Laplacian eigenmaps
    """
    
    def __init__(self, dimensions: int = 128):
        """
        Initialize spectral embeddings
        
        Args:
            dimensions: Number of embedding dimensions
        """
        self.dimensions = dimensions
        self.embeddings = None
        
    def fit(self, graph: nx.Graph) -> pd.DataFrame:
        """
        Compute spectral embeddings using normalized Laplacian
        
        Args:
            graph: NetworkX graph
            
        Returns:
            DataFrame with spectral embeddings
        """
        # Convert to undirected
        if isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
            G = nx.Graph(graph)
        else:
            G = graph
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        # Compute normalized Laplacian
        L = nx.normalized_laplacian_matrix(G)
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = eigsh(
            L.astype(np.float64), 
            k=min(self.dimensions + 1, n_nodes - 1),
            which='SM'
        )
        
        # Skip first eigenvector (constant)
        self.embeddings = eigenvectors[:, 1:self.dimensions + 1]
        
        # Create DataFrame
        embedding_df = pd.DataFrame(
            self.embeddings,
            index=nodes,
            columns=[f'spectral_{i}' for i in range(self.dimensions)]
        )
        
        return embedding_df
    
    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific nodes"""
        if self.embeddings is None:
            raise ValueError("Model not fitted yet")
        
        if isinstance(self.embeddings, pd.DataFrame):
            return self.embeddings.loc[node_ids].values
        
        return self.embeddings


class StructuralEmbeddings:
    """
    Graph structural embeddings based on node properties
    Includes degree-based, centrality-based, and role-based embeddings
    """
    
    def __init__(self):
        self.embeddings = None
        
    def fit(self, graph: nx.Graph) -> pd.DataFrame:
        """
        Compute structural embeddings
        
        Args:
            graph: NetworkX graph
            
        Returns:
            DataFrame with structural embeddings
        """
        nodes = list(graph.nodes())
        
        features = []
        
        # Degree features
        degrees = dict(graph.degree())
        features.append(pd.Series(degrees, name='degree'))
        
        # Clustering coefficient
        if isinstance(graph, nx.Graph):
            clustering = nx.clustering(graph)
            features.append(pd.Series(clustering, name='clustering'))
        
        # Core number (k-core decomposition)
        try:
            core_numbers = nx.core_number(graph)
            features.append(pd.Series(core_numbers, name='core_number'))
        except:
            pass
        
        # Create feature matrix
        feature_df = pd.concat(features, axis=1)
        feature_df = feature_df.fillna(0)
        
        # Apply PCA for dimensionality reduction and decorrelation
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        
        pca = PCA(n_components=min(16, len(features)))
        reduced_features = pca.fit_transform(scaled_features)
        
        # Create embedding DataFrame
        embedding_df = pd.DataFrame(
            reduced_features,
            index=nodes,
            columns=[f'structural_{i}' for i in range(reduced_features.shape[1])]
        )
        
        self.embeddings = embedding_df
        return embedding_df
    
    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific nodes"""
        if self.embeddings is None:
            raise ValueError("Model not fitted yet")
        return self.embeddings.loc[node_ids].values


class MetapathEmbeddings:
    """
    Metapath-based embeddings for heterogeneous graphs
    Captures semantic relationships through predefined paths
    """
    
    def __init__(self, metapaths: List[List[str]] = None):
        """
        Initialize metapath embeddings
        
        Args:
            metapaths: List of metapaths, each as list of node types
                      e.g., [['Provider', 'Claim', 'Patient', 'Claim', 'Provider']]
        """
        if metapaths is None:
            self.metapaths = [
                ['Provider', 'Claim', 'Provider'],  # Co-billing
                ['Provider', 'Patient', 'Provider'],  # Shared patients
                ['Provider', 'BankAccount', 'Provider'],  # Shared bank
                ['Provider', 'Exclusion', 'Provider'],  # Co-exclusion
            ]
        else:
            self.metapaths = metapaths
        
        self.embeddings = None
        self.metapath_graphs = {}
    
    def _extract_metapath_graph(self, graph: nx.MultiDiGraph, 
                                metapath: List[str]) -> nx.Graph:
        """
        Extract metapath-induced graph
        
        Args:
            graph: Heterogeneous graph
            metapath: List of node types defining the path
            
        Returns:
            Homogeneous graph connecting source-type nodes via metapath
        """
        # Get all nodes of the starting type
        source_type = metapath[0]
        source_nodes = [
            node for node, attr in graph.nodes(data=True)
            if attr.get('node_type') == source_type
        ]
        
        # Create metapath graph
        mp_graph = nx.Graph()
        mp_graph.add_nodes_from(source_nodes)
        
        # Find all paths matching the metapath
        for source in source_nodes:
            # Simple BFS to find metapath instances
            paths = self._find_metapath_instances(graph, source, metapath)
            
            for target in paths:
                if target != source:
                    if mp_graph.has_edge(source, target):
                        mp_graph[source][target]['weight'] += 1
                    else:
                        mp_graph.add_edge(source, target, weight=1)
        
        return mp_graph
    
    def _find_metapath_instances(self, graph: nx.MultiDiGraph, 
                                 start_node: str, 
                                 metapath: List[str]) -> List[str]:
        """
        Find all nodes reachable via the metapath
        
        Args:
            graph: Heterogeneous graph
            start_node: Starting node
            metapath: Node type sequence
            
        Returns:
            List of end nodes matching the metapath
        """
        current_nodes = {start_node}
        
        for step_idx, node_type in enumerate(metapath[1:], 1):
            next_nodes = set()
            
            for current in current_nodes:
                for neighbor in graph.neighbors(current):
                    if graph.nodes[neighbor].get('node_type') == node_type:
                        next_nodes.add(neighbor)
            
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        return list(current_nodes)
    
    def fit(self, graph: nx.MultiDiGraph) -> pd.DataFrame:
        """
        Generate metapath embeddings
        
        Args:
            graph: Heterogeneous graph
            
        Returns:
            DataFrame with metapath embeddings
        """
        # Get provider nodes
        providers = [
            node for node, attr in graph.nodes(data=True)
            if attr.get('node_type') in ['Provider', 'Provider_Ind', 'Provider_Org']
        ]
        
        all_embeddings = []
        
        for metapath in self.metapaths:
            # Extract metapath graph
            mp_graph = self._extract_metapath_graph(graph, metapath)
            self.metapath_graphs['-'.join(metapath)] = mp_graph
            
            # Compute embeddings using Node2Vec
            node2vec = Node2VecEmbeddings(
                dimensions=16,  # Smaller dimension per metapath
                walk_length=10,
                num_walks=50
            )
            
            try:
                embeddings = node2vec.fit(mp_graph)
                all_embeddings.append(embeddings)
            except Exception as e:
                print(f"Failed to compute embeddings for metapath {metapath}: {e}")
                # Create zero embeddings
                zero_emb = pd.DataFrame(
                    0,
                    index=providers,
                    columns=[f'mp_{len(all_embeddings)}_{i}' for i in range(16)]
                )
                all_embeddings.append(zero_emb)
        
        # Combine all metapath embeddings
        combined_embeddings = pd.concat(all_embeddings, axis=1)
        
        # Align with provider list
        combined_embeddings = combined_embeddings.reindex(providers).fillna(0)
        
        self.embeddings = combined_embeddings
        return combined_embeddings
    
    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific nodes"""
        if self.embeddings is None:
            raise ValueError("Model not fitted yet")
        return self.embeddings.loc[node_ids].values


class GraphAutoencoderEmbeddings:
    """
    Graph autoencoder for unsupervised embedding learning
    Uses adjacency matrix reconstruction as pretext task
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 encoding_dim: int = 64,
                 learning_rate: float = 0.01,
                 epochs: int = 100):
        """
        Initialize graph autoencoder
        
        Args:
            hidden_dim: Hidden layer dimension
            encoding_dim: Bottleneck/encoding dimension
            learning_rate: Learning rate
            epochs: Training epochs
        """
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embeddings = None
        self.encoder_weights = None
        self.decoder_weights = None
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _initialize_weights(self, input_dim: int):
        """Initialize autoencoder weights"""
        scale = np.sqrt(2.0 / input_dim)
        
        self.encoder_weights = {
            'W1': np.random.randn(input_dim, self.hidden_dim) * scale,
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, self.encoding_dim) * scale,
            'b2': np.zeros(self.encoding_dim)
        }
        
        self.decoder_weights = {
            'W1': np.random.randn(self.encoding_dim, self.hidden_dim) * scale,
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, input_dim) * scale,
            'b2': np.zeros(input_dim)
        }
    
    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through autoencoder
        
        Args:
            X: Input features (adjacency matrix rows)
            
        Returns:
            Tuple of (reconstructed, encoded)
        """
        # Encoder
        hidden1 = self._sigmoid(X @ self.encoder_weights['W1'] + self.encoder_weights['b1'])
        encoded = self._sigmoid(hidden1 @ self.encoder_weights['W2'] + self.encoder_weights['b2'])
        
        # Decoder
        hidden2 = self._sigmoid(encoded @ self.decoder_weights['W1'] + self.decoder_weights['b1'])
        reconstructed = self._sigmoid(hidden2 @ self.decoder_weights['W2'] + self.decoder_weights['b2'])
        
        return reconstructed, encoded
    
    def fit(self, graph: nx.Graph) -> pd.DataFrame:
        """
        Train graph autoencoder and generate embeddings
        
        Args:
            graph: NetworkX graph
            
        Returns:
            DataFrame with autoencoder embeddings
        """
        # Get adjacency matrix
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        A = nx.adjacency_matrix(graph).toarray()
        
        # Initialize weights
        self._initialize_weights(n_nodes)
        
        # Training
        print("Training graph autoencoder...")
        for epoch in range(self.epochs):
            # Forward pass
            reconstructed, encoded = self._forward(A)
            
            # Reconstruction loss (binary cross-entropy)
            loss = -np.mean(
                A * np.log(reconstructed + 1e-10) + 
                (1 - A) * np.log(1 - reconstructed + 1e-10)
            )
            
            # Backward pass (simplified - using gradient of BCE)
            d_reconstructed = (reconstructed - A) / (reconstructed * (1 - reconstructed) + 1e-10) / n_nodes
            
            # Decoder gradients
            d_hidden2 = d_reconstructed @ self.decoder_weights['W2'].T
            d_hidden2 *= (hidden2 := self._sigmoid(encoded @ self.decoder_weights['W1'] + self.decoder_weights['b1'])) * (1 - hidden2)
            
            # Encoder gradients
            d_encoded = d_hidden2 @ self.decoder_weights['W1'].T
            d_encoded *= encoded * (1 - encoded)
            
            # Update weights (simple SGD)
            lr = self.learning_rate * (0.95 ** epoch)  # Learning rate decay
            
            # Update decoder weights
            self.decoder_weights['W2'] -= lr * hidden2.T @ d_reconstructed
            self.decoder_weights['b2'] -= lr * d_reconstructed.sum(axis=0)
            self.decoder_weights['W1'] -= lr * encoded.T @ d_hidden2
            self.decoder_weights['b1'] -= lr * d_hidden2.sum(axis=0)
            
            # Update encoder weights
            self.encoder_weights['W2'] -= lr * A.T @ d_encoded
            self.encoder_weights['b2'] -= lr * d_encoded.sum(axis=0)
            self.encoder_weights['W1'] -= lr * A.T @ d_encoded @ self.encoder_weights['W2'].T * A * (1 - A)
            self.encoder_weights['b1'] -= lr * (d_encoded @ self.encoder_weights['W2'].T * A * (1 - A)).sum(axis=0)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        # Get final embeddings
        _, final_encoded = self._forward(A)
        
        self.embeddings = pd.DataFrame(
            final_encoded,
            index=nodes,
            columns=[f'autoencoder_{i}' for i in range(self.encoding_dim)]
        )
        
        return self.embeddings
    
    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific nodes"""
        if self.embeddings is None:
            raise ValueError("Model not fitted yet")
        return self.embeddings.loc[node_ids].values


class EmbeddingEnsemble:
    """
    Ensemble of multiple embedding methods
    """
    
    def __init__(self, methods: List[str] = None):
        """
        Initialize embedding ensemble
        
        Args:
            methods: List of embedding methods to use
                    Options: 'node2vec', 'spectral', 'structural', 'metapath', 'autoencoder'
        """
        if methods is None:
            self.methods = ['node2vec', 'spectral', 'structural']
        else:
            self.methods = methods
        
        self.embeddings = {}
        self.combined_embeddings = None
        
    def fit(self, graph: nx.Graph) -> pd.DataFrame:
        """
        Generate embeddings using all specified methods
        
        Args:
            graph: NetworkX graph (can be heterogeneous for metapath)
            
        Returns:
            DataFrame with combined embeddings
        """
        embedding_dfs = []
        
        if 'node2vec' in self.methods:
            print("Computing Node2Vec embeddings...")
            node2vec = Node2VecEmbeddings(dimensions=32)
            try:
                node2vec_emb = node2vec.fit(graph)
                embedding_dfs.append(node2vec_emb)
                print(f"  Node2Vec: {node2vec_emb.shape[1]} dimensions")
            except Exception as e:
                print(f"  Node2Vec failed: {e}")
        
        if 'spectral' in self.methods:
            print("Computing Spectral embeddings...")
            spectral = SpectralEmbeddings(dimensions=32)
            try:
                spectral_emb = spectral.fit(graph)
                embedding_dfs.append(spectral_emb)
                print(f"  Spectral: {spectral_emb.shape[1]} dimensions")
            except Exception as e:
                print(f"  Spectral failed: {e}")
        
        if 'structural' in self.methods:
            print("Computing Structural embeddings...")
            structural = StructuralEmbeddings()
            try:
                structural_emb = structural.fit(graph)
                embedding_dfs.append(structural_emb)
                print(f"  Structural: {structural_emb.shape[1]} dimensions")
            except Exception as e:
                print(f"  Structural failed: {e}")
        
        if 'metapath' in self.methods:
            print("Computing Metapath embeddings...")
            metapath = MetapathEmbeddings()
            try:
                metapath_emb = metapath.fit(graph)
                embedding_dfs.append(metapath_emb)
                print(f"  Metapath: {metapath_emb.shape[1]} dimensions")
            except Exception as e:
                print(f"  Metapath failed: {e}")
        
        if 'autoencoder' in self.methods:
            print("Computing Graph Autoencoder embeddings...")
            autoencoder = GraphAutoencoderEmbeddings(
                hidden_dim=64,
                encoding_dim=32,
                epochs=50
            )
            try:
                autoencoder_emb = autoencoder.fit(graph)
                embedding_dfs.append(autoencoder_emb)
                print(f"  Autoencoder: {autoencoder_emb.shape[1]} dimensions")
            except Exception as e:
                print(f"  Autoencoder failed: {e}")
        
        # Combine all embeddings
        if embedding_dfs:
            self.combined_embeddings = pd.concat(embedding_dfs, axis=1)
            print(f"\nTotal embedding dimensions: {self.combined_embeddings.shape[1]}")
        else:
            print("Warning: No embeddings were successfully computed")
            nodes = list(graph.nodes())
            self.combined_embeddings = pd.DataFrame(
                np.zeros((len(nodes), 10)),
                index=nodes
            )
        
        return self.combined_embeddings
    
    def get_embeddings(self, node_ids: List[str]) -> np.ndarray:
        """Get combined embeddings for specific nodes"""
        if self.combined_embeddings is None:
            raise ValueError("Model not fitted yet")
        return self.combined_embeddings.loc[node_ids].values