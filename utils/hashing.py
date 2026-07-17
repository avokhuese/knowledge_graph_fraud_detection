"""
Privacy-preserving hashing utilities for sensitive data
"""
import hashlib
import hmac
import pandas as pd
import numpy as np
from typing import Union, List, Optional

class DataHasher:
    """Handles hashing of sensitive fields to protect PII/PHI"""
    
    def __init__(self, salt: str = None):
        """
        Initialize hasher with a salt
        
        Args:
            salt: Secret salt for HMAC. In production, load from secure vault
        """
        self.salt = salt or "production-salt-change-me"
        self.salt_bytes = self.salt.encode('utf-8')
    
    def hash_value(self, value: Union[str, int, float]) -> Optional[str]:
        """
        Hash a single value using HMAC-SHA256
        
        Args:
            value: Value to hash
            
        Returns:
            Hex-encoded hash string, or None if input is None/NaN
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        
        # Convert to string if needed
        if not isinstance(value, str):
            value = str(value)
        
        # Empty string check
        if value.strip() == '':
            return None
        
        # Create HMAC hash
        h = hmac.new(
            self.salt_bytes,
            value.encode('utf-8'),
            hashlib.sha256
        )
        return h.hexdigest()
    
    def hash_series(self, series: pd.Series) -> pd.Series:
        """
        Hash all values in a pandas Series
        
        Args:
            series: Pandas Series to hash
            
        Returns:
            Series of hashed values
        """
        return series.apply(self.hash_value)
    
    def hash_dataframe_columns(self, 
                               df: pd.DataFrame, 
                               columns: List[str]) -> pd.DataFrame:
        """
        Hash specific columns in a DataFrame
        
        Args:
            df: Input DataFrame
            columns: List of column names to hash
            
        Returns:
            DataFrame with hashed columns
        """
        df_hashed = df.copy()
        for col in columns:
            if col in df_hashed.columns:
                df_hashed[f'{col}_hash'] = self.hash_series(df_hashed[col])
                # Optionally drop original column
                # df_hashed = df_hashed.drop(columns=[col])
        return df_hashed
    
    def create_entity_id(self, *args) -> str:
        """
        Create a deterministic entity ID by hashing multiple fields together
        
        Args:
            *args: Fields to combine and hash
            
        Returns:
            Deterministic hash-based ID
        """
        combined = '|'.join([str(arg) for arg in args if arg is not None])
        return self.hash_value(combined)
    
    def verify_hash(self, value: str, hash_value: str) -> bool:
        """
        Verify if a value matches a given hash
        
        Args:
            value: Original value
            hash_value: Previously computed hash
            
        Returns:
            True if value hashes to the same hash
        """
        computed_hash = self.hash_value(value)
        return hmac.compare_digest(computed_hash, hash_value)


class TokenGenerator:
    """Generate secure tokens for anonymized IDs"""
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate a random token
        
        Args:
            length: Number of random bytes
            
        Returns:
            Hex-encoded random token
        """
        import secrets
        return secrets.token_hex(length)
    
    @staticmethod
    def create_lookup_table(original_ids: pd.Series) -> pd.DataFrame:
        """
        Create a mapping table from original IDs to tokens
        
        Args:
            original_ids: Original identifier series
            
        Returns:
            DataFrame with mapping
        """
        unique_ids = original_ids.unique()
        tokens = [TokenGenerator.generate_token() for _ in range(len(unique_ids))]
        
        lookup = pd.DataFrame({
            'original_id': unique_ids,
            'token': tokens
        })
        
        return lookup