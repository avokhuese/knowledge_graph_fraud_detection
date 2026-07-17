import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Tuple, Optional

class FraudGNN(nn.Module):
    """Graph Neural Network for fraud detection"""
    
    def __init__(self, 
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 node_features: Dict[str, int],
                 hidden_channels: int = 64,
                 output_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout
        
        # Input projections for each node type
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            in_dim = node_features.get(node_type, 1)
            self.input_projections[node_type] = nn.Linear(in_dim, hidden_channels)
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    hidden_channels, 
                    hidden_channels
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_channels, output_dim)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(hidden_channels)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])
        
    def forward(self, x_dict, edge_index_dict):
        # Input projection
        x_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Heterogeneous convolutions
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: batch_norm[node_type](x)
                for node_type, x in x_dict.items()
            }
            x_dict = {
                node_type: F.relu(x)
                for node_type, x in x_dict.items()
            }
            x_dict = {
                node_type: F.dropout(x, p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }
        
        # Output projection
        embeddings = {}
        for node_type, x in x_dict.items():
            embeddings[node_type] = self.output_projection(x)
        
        return embeddings


class LinkPredictionPretraining(nn.Module):
    """Self-supervised pretraining via link prediction"""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, emb_u, emb_v):
        # Concatenate embeddings
        combined = torch.cat([emb_u, emb_v], dim=1)
        return self.link_predictor(combined)


class GNNTrainer:
    """Handles training of GNN models"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def pretrain_link_prediction(self, 
                                 data: HeteroData,
                                 epochs: int = 100,
                                 lr: float = 0.001) -> FraudGNN:
        """Pretrain GNN using link prediction on all edge types"""
        
        link_predictor = LinkPredictionPretraining().to(self.device)
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(link_predictor.parameters()),
            lr=lr
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                
                # Positive samples
                pos_embeddings = self.model(data.x_dict, data.edge_index_dict)
                src_type, _, dst_type = edge_type
                
                pos_src_emb = pos_embeddings[src_type][edge_index[0]]
                pos_dst_emb = pos_embeddings[dst_type][edge_index[1]]
                pos_pred = link_predictor(pos_src_emb, pos_dst_emb)
                
                # Negative samples
                neg_dst = torch.randint(0, data[dst_type].num_nodes, edge_index[1].shape)
                neg_dst_emb = pos_embeddings[dst_type][neg_dst]
                neg_pred = link_predictor(pos_src_emb, neg_dst_emb)
                
                # Loss
                pos_loss = F.binary_cross_entropy(
                    pos_pred, 
                    torch.ones_like(pos_pred)
                )
                neg_loss = F.binary_cross_entropy(
                    neg_pred,
                    torch.zeros_like(neg_pred)
                )
                loss = (pos_loss + neg_loss) / 2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
        
        return self.model
    
    def train_supervised(self,
                        data: HeteroData,
                        labels: Dict[str, torch.Tensor],
                        train_mask: Dict[str, torch.Tensor],
                        val_mask: Optional[Dict[str, torch.Tensor]] = None,
                        epochs: int = 100,
                        lr: float = 0.001) -> Dict:
        """Supervised training for fraud classification"""
        
        # Classification head
        classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(classifier.parameters()),
            lr=lr
        )
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            
            # Get embeddings
            embeddings = self.model(data.x_dict, data.edge_index_dict)
            
            # Only classify providers
            provider_emb = embeddings['Provider']
            predictions = classifier(provider_emb).squeeze()
            
            # Focal loss for class imbalance
            alpha = 0.25
            gamma = 2.0
            
            train_predictions = predictions[train_mask['Provider']]
            train_labels = labels['Provider'][train_mask['Provider']]
            
            pt = torch.where(train_labels == 1, train_predictions, 1 - train_predictions)
            focal_weight = (1 - pt) ** gamma
            
            bce_loss = F.binary_cross_entropy(
                train_predictions, 
                train_labels.float(),
                reduction='none'
            )
            focal_loss = (focal_weight * bce_loss).mean()
            
            # Add alpha weighting for positive class
            alpha_weight = torch.where(
                train_labels == 1, 
                alpha, 
                1 - alpha
            )
            focal_loss = (alpha_weight * focal_weight * bce_loss).mean()
            
            optimizer.zero_grad()
            focal_loss.backward()
            optimizer.step()
            
            history['train_loss'].append(focal_loss.item())
            
            # Validation
            if val_mask is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = predictions[val_mask['Provider']]
                    val_labels = labels['Provider'][val_mask['Provider']]
                    val_loss = F.binary_cross_entropy(
                        val_predictions, 
                        val_labels.float()
                    )
                    history['val_loss'].append(val_loss.item())
            
            if epoch % 10 == 0:
                msg = f"Epoch {epoch}: Train Loss = {focal_loss.item():.4f}"
                if val_mask is not None:
                    msg += f", Val Loss = {val_loss.item():.4f}"
                print(msg)
        
        return history
    
    def extract_embeddings(self, data: HeteroData) -> Dict[str, np.ndarray]:
        """Extract node embeddings for downstream use"""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data.x_dict, data.edge_index_dict)
            return {
                node_type: emb.cpu().numpy()
                for node_type, emb in embeddings.items()
            }