"""
Data preprocessing and feature engineering for fraud detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ClaimsPreprocessor:
    """Preprocess claims data for analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def clean_claims_data(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize claims data
        
        Args:
            claims: Raw claims DataFrame
            
        Returns:
            Cleaned claims DataFrame
        """
        df = claims.copy()
        
        # Standardize date columns
        date_columns = ['date_of_service', 'claim_received_date', 'adjudication_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean monetary columns
        monetary_cols = ['billed_amount', 'paid_amount', 'allowed_amount']
        for col in monetary_cols:
            if col in df.columns:
                # Remove dollar signs and commas
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Clip negative values
                df[col] = df[col].clip(lower=0)
        
        # Clean code fields
        code_cols = ['procedure_code', 'diagnosis_code', 'modifier']
        for col in code_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                # Remove decimal points from CPT codes
                df[col] = df[col].str.replace('.', '', regex=False)
        
        # Standardize provider NPI
        if 'provider_npi' in df.columns:
            df['provider_npi'] = df['provider_npi'].astype(str).str.strip()
        
        return df
    
    def extract_temporal_features(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from claims
        
        Args:
            claims: Claims DataFrame with date_of_service
            
        Returns:
            DataFrame with temporal features per provider
        """
        df = claims.copy()
        
        if 'date_of_service' not in df.columns:
            return pd.DataFrame()
        
        df['date_of_service'] = pd.to_datetime(df['date_of_service'])
        
        # Add temporal components
        df['day_of_week'] = df['date_of_service'].dt.dayofweek
        df['month'] = df['date_of_service'].dt.month
        df['quarter'] = df['date_of_service'].dt.quarter
        df['year'] = df['date_of_service'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Per-provider temporal features
        provider_features = []
        
        for npi, group in df.groupby('provider_npi'):
            features = {'provider_npi': npi}
            
            # Date range
            date_range = (group['date_of_service'].max() - 
                         group['date_of_service'].min()).days
            features['active_days'] = max(date_range, 1)
            
            # Claims per day
            features['claims_per_day'] = len(group) / features['active_days']
            
            # Weekend billing ratio
            if len(group) > 0:
                features['weekend_ratio'] = group['is_weekend'].mean()
            
            # Monthly variation
            monthly_counts = group.groupby('month').size()
            if len(monthly_counts) > 1:
                features['monthly_variation'] = monthly_counts.std() / max(monthly_counts.mean(), 1)
            else:
                features['monthly_variation'] = 0
            
            # Billing velocity (recent vs overall)
            median_date = group['date_of_service'].median()
            recent = group[group['date_of_service'] >= median_date]
            older = group[group['date_of_service'] < median_date]
            
            if len(older) > 0:
                recent_daily = len(recent) / max((recent['date_of_service'].max() - 
                                                   recent['date_of_service'].min()).days, 1)
                older_daily = len(older) / max((older['date_of_service'].max() - 
                                                older['date_of_service'].min()).days, 1)
                features['velocity_change'] = (recent_daily - older_daily) / max(older_daily, 0.001)
            else:
                features['velocity_change'] = 0
            
            provider_features.append(features)
        
        return pd.DataFrame(provider_features)
    
    def extract_billing_features(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Extract billing pattern features
        
        Args:
            claims: Claims DataFrame
            
        Returns:
            DataFrame with billing features per provider
        """
        provider_features = []
        
        for npi, group in claims.groupby('provider_npi'):
            features = {'provider_npi': npi}
            
            # Volume features
            features['total_claims'] = len(group)
            features['total_billed'] = group['billed_amount'].sum() if 'billed_amount' in group.columns else 0
            features['total_paid'] = group['paid_amount'].sum() if 'paid_amount' in group.columns else 0
            
            # Average amounts
            if 'billed_amount' in group.columns and len(group) > 0:
                features['avg_billed'] = group['billed_amount'].mean()
                features['std_billed'] = group['billed_amount'].std()
                features['max_billed'] = group['billed_amount'].max()
                
                # High-value claim ratio
                threshold = group['billed_amount'].quantile(0.95)
                features['high_value_ratio'] = (group['billed_amount'] > threshold).mean()
            
            # Payment ratio
            if 'paid_amount' in group.columns and 'billed_amount' in group.columns:
                total_billed = group['billed_amount'].sum()
                if total_billed > 0:
                    features['payment_ratio'] = group['paid_amount'].sum() / total_billed
                else:
                    features['payment_ratio'] = 0
            
            # Patient features
            if 'beneficiary_hash' in group.columns:
                features['unique_patients'] = group['beneficiary_hash'].nunique()
                features['claims_per_patient'] = len(group) / max(features['unique_patients'], 1)
            
            # Procedure code diversity
            if 'procedure_code' in group.columns:
                features['unique_procedures'] = group['procedure_code'].nunique()
                top_procedure = group['procedure_code'].value_counts().iloc[0]
                features['top_procedure_concentration'] = top_procedure / len(group)
            
            provider_features.append(features)
        
        return pd.DataFrame(provider_features)
    
    def detect_upcoding_patterns(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential upcoding patterns
        
        Args:
            claims: Claims DataFrame with procedure codes
            
        Returns:
            DataFrame with upcoding indicators per provider
        """
        if 'procedure_code' not in claims.columns:
            return pd.DataFrame()
        
        # Define E/M code hierarchy (simplified)
        em_codes = {
            '99201': 1, '99202': 2, '99203': 3, '99204': 4, '99205': 5,  # New patient
            '99211': 1, '99212': 2, '99213': 3, '99214': 4, '99215': 5,  # Established patient
            '99281': 1, '99282': 2, '99283': 3, '99284': 4, '99285': 5,  # Emergency
        }
        
        provider_features = []
        
        for npi, group in claims.groupby('provider_npi'):
            features = {'provider_npi': npi}
            
            # Filter for E/M codes
            em_claims = group[group['procedure_code'].isin(em_codes.keys())]
            
            if len(em_claims) > 0:
                # Calculate complexity scores
                em_claims = em_claims.copy()
                em_claims['complexity'] = em_claims['procedure_code'].map(em_codes)
                
                features['avg_complexity'] = em_claims['complexity'].mean()
                features['high_complexity_ratio'] = (em_claims['complexity'] >= 4).mean()
                
                # Detect level 5 concentration
                level_5_codes = [code for code, level in em_codes.items() if level == 5]
                features['level_5_ratio'] = em_claims['procedure_code'].isin(level_5_codes).mean()
            else:
                features['avg_complexity'] = 0
                features['high_complexity_ratio'] = 0
                features['level_5_ratio'] = 0
            
            provider_features.append(features)
        
        return pd.DataFrame(provider_features)
    
    def detect_impossible_days(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Detect claims suggesting impossible work hours
        
        Args:
            claims: Claims DataFrame
            
        Returns:
            DataFrame with impossible day indicators
        """
        if 'procedure_code' not in claims.columns:
            return pd.DataFrame()
        
        # Time units for common procedures (simplified)
        procedure_times = {
            '99213': 15,  # 15 minutes
            '99214': 25,  # 25 minutes
            '99215': 40,  # 40 minutes
            '90837': 60,  # 60 minutes psychotherapy
            '97110': 15,  # 15 minutes therapeutic exercise
        }
        
        provider_features = []
        
        for npi, group in claims.groupby('provider_npi'):
            features = {'provider_npi': npi}
            
            if 'date_of_service' in group.columns:
                # Calculate total minutes per day
                group = group.copy()
                group['date'] = pd.to_datetime(group['date_of_service']).dt.date
                group['time_units'] = group['procedure_code'].map(procedure_times).fillna(15)
                
                daily_minutes = group.groupby('date')['time_units'].sum()
                
                # Flag days exceeding 24 hours (1440 minutes)
                impossible_days = (daily_minutes > 1440).sum()
                features['impossible_days_count'] = impossible_days
                features['impossible_days_ratio'] = impossible_days / max(len(daily_minutes), 1)
                
                # Average daily hours
                features['avg_daily_hours'] = daily_minutes.mean() / 60
                features['max_daily_hours'] = daily_minutes.max() / 60
            else:
                features['impossible_days_count'] = 0
                features['impossible_days_ratio'] = 0
                features['avg_daily_hours'] = 0
                features['max_daily_hours'] = 0
            
            provider_features.append(features)
        
        return pd.DataFrame(provider_features)


class FeatureAggregator:
    """Aggregate all features for model training"""
    
    def __init__(self):
        self.preprocessor = ClaimsPreprocessor()
    
    def compute_all_features(self, 
                            claims: pd.DataFrame, 
                            providers: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all claims-behavioral features
        
        Args:
            claims: Claims DataFrame
            providers: Providers DataFrame
            
        Returns:
            DataFrame with all features, indexed by provider NPI
        """
        # Clean claims
        claims_clean = self.preprocessor.clean_claims_data(claims)
        
        # Extract features
        temporal_features = self.preprocessor.extract_temporal_features(claims_clean)
        billing_features = self.preprocessor.extract_billing_features(claims_clean)
        upcoding_features = self.preprocessor.detect_upcoding_patterns(claims_clean)
        impossible_days = self.preprocessor.detect_impossible_days(claims_clean)
        
        # Merge all features
        feature_dfs = [temporal_features, billing_features, 
                      upcoding_features, impossible_days]
        
        all_features = providers[['npi']].copy()
        all_features = all_features.rename(columns={'npi': 'provider_npi'})
        
        for df in feature_dfs:
            if not df.empty and 'provider_npi' in df.columns:
                all_features = all_features.merge(df, on='provider_npi', how='left')
        
        # Set index
        all_features = all_features.set_index('provider_npi')
        
        # Fill missing values
        all_features = all_features.fillna(0)
        
        # Handle infinite values
        all_features = all_features.replace([np.inf, -np.inf], 0)
        
        return all_features