"""
Data validation utilities for fraud detection pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates input data for fraud detection pipeline"""
    
    @staticmethod
    def validate_providers(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate provider data
        
        Args:
            df: Provider DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        required_cols = ['npi']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if 'npi' in df.columns:
            # Check for missing NPIs
            missing = df['npi'].isna().sum()
            if missing > 0:
                errors.append(f"Found {missing} missing NPI values")
            
            # Check for duplicates
            duplicates = df['npi'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate NPIs")
            
            # Validate NPI format (10 digits)
            invalid_npis = df['npi'].apply(
                lambda x: not (isinstance(x, str) and len(x) == 10 and x.isdigit())
            )
            if invalid_npis.sum() > 0:
                errors.append(f"Found {invalid_npis.sum()} invalid NPI formats")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_claims(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate claims data
        
        Args:
            df: Claims DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        required_cols = ['claim_id', 'provider_npi', 'date_of_service']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if 'date_of_service' in df.columns:
            # Check date format
            try:
                pd.to_datetime(df['date_of_service'])
            except:
                errors.append("Invalid date format in date_of_service")
            
            # Check for future dates
            future_dates = pd.to_datetime(df['date_of_service']) > datetime.now()
            if future_dates.sum() > 0:
                errors.append(f"Found {future_dates.sum()} claims with future dates")
        
        if 'billed_amount' in df.columns:
            # Check for negative amounts
            negative = df['billed_amount'] < 0
            if negative.sum() > 0:
                errors.append(f"Found {negative.sum()} claims with negative billed amounts")
            
            # Check for unreasonably high amounts (e.g., > $1M)
            high_amounts = df['billed_amount'] > 1_000_000
            if high_amounts.sum() > 0:
                logger.warning(f"Found {high_amounts.sum()} claims with amount > $1M")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_exclusions(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate exclusion list data
        
        Args:
            df: Exclusions DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        required_cols = ['entity_npi', 'exclusion_date']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def check_temporal_consistency(claims: pd.DataFrame, 
                                   exclusions: pd.DataFrame) -> Dict:
        """
        Check temporal consistency between claims and exclusions
        
        Args:
            claims: Claims DataFrame
            exclusions: Exclusions DataFrame
            
        Returns:
            Dictionary with consistency metrics
        """
        consistency = {
            'claims_date_range': None,
            'exclusions_date_range': None,
            'potential_leakage': False
        }
        
        if 'date_of_service' in claims.columns:
            claims['date_of_service'] = pd.to_datetime(claims['date_of_service'])
            consistency['claims_date_range'] = {
                'min': claims['date_of_service'].min(),
                'max': claims['date_of_service'].max()
            }
        
        if 'exclusion_date' in exclusions.columns:
            exclusions['exclusion_date'] = pd.to_datetime(exclusions['exclusion_date'])
            consistency['exclusions_date_range'] = {
                'min': exclusions['exclusion_date'].min(),
                'max': exclusions['exclusion_date'].max()
            }
        
        # Check for potential temporal leakage
        if (consistency['claims_date_range'] and 
            consistency['exclusions_date_range']):
            if (consistency['claims_date_range']['max'] > 
                consistency['exclusions_date_range']['min']):
                consistency['potential_leakage'] = True
        
        return consistency
    
    @staticmethod
    def generate_validation_report(providers: pd.DataFrame,
                                   claims: pd.DataFrame,
                                   exclusions: pd.DataFrame) -> Dict:
        """
        Generate comprehensive validation report
        
        Returns:
            Dictionary with validation results
        """
        report = {}
        
        # Validate each dataset
        for name, df, validator in [
            ('providers', providers, DataValidator.validate_providers),
            ('claims', claims, DataValidator.validate_claims),
            ('exclusions', exclusions, DataValidator.validate_exclusions)
        ]:
            is_valid, errors = validator(df)
            report[name] = {
                'is_valid': is_valid,
                'num_records': len(df),
                'errors': errors
            }
        
        # Data statistics
        report['statistics'] = {
            'total_providers': len(providers),
            'total_claims': len(claims),
            'total_exclusions': len(exclusions),
            'unique_providers_in_claims': claims['provider_npi'].nunique() if 'provider_npi' in claims.columns else 0,
            'providers_without_claims': len(set(providers['npi']) - set(claims['provider_npi'])) if 'npi' in providers.columns and 'provider_npi' in claims.columns else 0
        }
        
        return report