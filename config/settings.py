import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class GraphConfig:
    """Graph construction configuration"""
    co_billing_window_days: int = 30  # Time window for CO_BILLS_WITH edges
    similarity_threshold: float = 0.8  # Cosine similarity threshold for SIMILAR_BILLING_PATTERN
    max_edge_types: List[str] = None
    
    def __post_init__(self):
        if self.max_edge_types is None:
            self.max_edge_types = [
                'BILLS_FOR', 'TREATS', 'REFERS_TO', 
                'CO_BILLS_WITH', 'SHARES_FACILITY', 
                'SHARES_BANK_ACCOUNT', 'OWNED_BY', 
                'FLAGGED_BY', 'SIMILAR_BILLING_PATTERN'
            ]

@dataclass
class ModelConfig:
    """Model training configuration"""
    embedding_dim: int = 128
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 256
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    data_dir: str = "./data"
    output_dir: str = "./output"
    model_dir: str = "./models"
    temporal_split_date: str = "2024-01-01"
    seed_providers_file: Optional[str] = None
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

class Settings:
    """Global settings"""
    graph = GraphConfig()
    model = ModelConfig()
    pipeline = PipelineConfig()
    
    # Privacy settings
    hash_salt: str = os.getenv("HASH_SALT", "default-salt-change-in-production")
    hash_algorithm: str = "sha256"
    
    # Evaluation settings
    investigator_capacity: int = 50  # Top K for precision@K/recall@K
    random_seed: int = 42