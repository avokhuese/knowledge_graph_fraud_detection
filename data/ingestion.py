import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import hmac
from pathlib import Path
from config.settings import Settings

class DataIngestion:
    """Handles loading and initial preprocessing of raw data"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.hash_salt = settings.hash_salt.encode()
        
    def hash_field(self, value: str) -> str:
        """Hash sensitive fields using HMAC-SHA256"""
        if pd.isna(value) or value == '':
            return None
        return hmac.new(
            self.hash_salt, 
            str(value).encode(), 
            hashlib.sha256
        ).hexdigest()
    
    def load_providers(self, filepath: str) -> pd.DataFrame:
        """Load provider data from NPPES or similar source"""
        df = pd.read_csv(filepath)
        
        # Required columns
        required_cols = ['npi', 'provider_type', 'taxonomy_code', 'specialty']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Hash sensitive fields
        df['provider_name_hash'] = df.get('provider_name', '').apply(self.hash_field)
        
        # Ensure NPI is string and clean
        df['npi'] = df['npi'].astype(str).str.strip()
        
        # Add node type
        df['node_type'] = np.where(
            df['provider_type'].isin(['Organization', 'Facility']), 
            'Provider_Org', 
            'Provider_Ind'
        )
        
        return df
    
    def load_claims(self, filepath: str) -> pd.DataFrame:
        """Load claims data"""
        df = pd.read_csv(filepath)
        
        required_cols = [
            'claim_id', 'provider_npi', 'beneficiary_id', 
            'date_of_service', 'billed_amount', 'paid_amount'
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Hash beneficiary IDs
        df['beneficiary_hash'] = df['beneficiary_id'].apply(self.hash_field)
        
        # Parse dates
        df['date_of_service'] = pd.to_datetime(df['date_of_service'])
        
        # Clean amounts
        for col in ['billed_amount', 'paid_amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def load_exclusions(self, filepath: str) -> pd.DataFrame:
        """Load OIG LEIE and other exclusion lists"""
        df = pd.read_csv(filepath)
        
        required_cols = ['entity_npi', 'exclusion_date', 'exclusion_source']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df['exclusion_date'] = pd.to_datetime(df['exclusion_date'])
        df['entity_npi'] = df['entity_npi'].astype(str).str.strip()
        
        # Add reinstatement date if available
        if 'reinstatement_date' in df.columns:
            df['reinstatement_date'] = pd.to_datetime(df['reinstatement_date'])
        else:
            df['reinstatement_date'] = pd.NaT
        
        return df
    
    def load_ownership(self, filepath: str) -> pd.DataFrame:
        """Load CMS-855 ownership disclosures"""
        df = pd.read_csv(filepath)
        
        required_cols = ['provider_npi', 'owner_id', 'ownership_percentage', 'owner_role']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Hash owner identities
        df['owner_hash'] = df['owner_id'].apply(self.hash_field)
        df['provider_npi'] = df['provider_npi'].astype(str).str.strip()
        
        return df