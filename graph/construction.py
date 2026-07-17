import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
from scipy.spatial.distance import cosine
from collections import defaultdict
import torch
from config.settings import Settings

class GraphBuilder:
    """Constructs the heterogeneous graph from claims and provider data"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.graph = nx.MultiDiGraph()
        self.node_attributes = defaultdict(dict)
        
    def build_graph(self, 
                    providers: pd.DataFrame,
                    claims: pd.DataFrame,
                    exclusions: Optional[pd.DataFrame] = None,
                    ownership: Optional[pd.DataFrame] = None,
                    bank_accounts: Optional[pd.DataFrame] = None) -> nx.MultiDiGraph:
        """
        Build the complete heterogeneous graph
        
        Args:
            providers: Provider dataframe with NPI and attributes
            claims: Claims dataframe
            exclusions: Exclusion records
            ownership: Ownership disclosures
            bank_accounts: Banking information
        """
        
        # Add provider nodes
        self._add_provider_nodes(providers)
        
        # Add claim nodes and edges
        if claims is not None:
            self._add_claim_nodes(claims)
            self._add_billing_edges(claims)
            self._add_treatment_edges(claims)
            self._add_referral_edges(claims)
            self._add_co_billing_edges(claims)
        
        # Add exclusion nodes and edges
        if exclusions is not None:
            self._add_exclusion_nodes(exclusions)
        
        # Add ownership edges
        if ownership is not None:
            self._add_ownership_edges(ownership)
        
        # Add facility sharing edges
        self._add_facility_edges(claims)
        
        # Add bank account sharing edges
        if bank_accounts is not None:
            self._add_bank_account_edges(bank_accounts)
        
        # Add billing similarity edges
        self._add_billing_similarity_edges(claims)
        
        return self.graph
    
    def _add_provider_nodes(self, providers: pd.DataFrame):
        """Add provider nodes with attributes"""
        for _, row in providers.iterrows():
            npi = row['npi']
            self.graph.add_node(
                npi,
                node_type='Provider',
                taxonomy=row.get('taxonomy_code', 'Unknown'),
                specialty=row.get('specialty', 'Unknown'),
                enrollment_date=row.get('enrollment_date', None),
                state=row.get('license_state', None)
            )
            
            # Store additional attributes for feature computation
            self.node_attributes[npi].update(row.to_dict())
    
    def _add_claim_nodes(self, claims: pd.DataFrame):
        """Add claim nodes"""
        for _, row in claims.iterrows():
            claim_id = f"CLAIM_{row['claim_id']}"
            self.graph.add_node(
                claim_id,
                node_type='Claim',
                date_of_service=row['date_of_service'],
                billed_amount=row['billed_amount'],
                paid_amount=row['paid_amount'],
                beneficiary_hash=row.get('beneficiary_hash')
            )
    
    def _add_billing_edges(self, claims: pd.DataFrame):
        """Add BILLS_FOR edges (Provider -> Claim)"""
        for _, row in claims.iterrows():
            self.graph.add_edge(
                row['provider_npi'],
                f"CLAIM_{row['claim_id']}",
                edge_type='BILLS_FOR',
                date=row['date_of_service'],
                amount=row['billed_amount']
            )
    
    def _add_treatment_edges(self, claims: pd.DataFrame):
        """Add TREATS edges (Provider -> Beneficiary)"""
        # Create beneficiary nodes
        beneficiary_mapping = {}
        
        for _, row in claims.iterrows():
            ben_hash = row.get('beneficiary_hash')
            if ben_hash:
                if ben_hash not in beneficiary_mapping:
                    self.graph.add_node(
                        ben_hash,
                        node_type='Beneficiary'
                    )
                    beneficiary_mapping[ben_hash] = True
                
                self.graph.add_edge(
                    row['provider_npi'],
                    ben_hash,
                    edge_type='TREATS',
                    date=row['date_of_service']
                )
    
    def _add_referral_edges(self, claims: pd.DataFrame):
        """Add REFERS_TO edges between providers"""
        referral_cols = ['referring_provider_npi', 'referring_npi']
        
        for col in referral_cols:
            if col in claims.columns:
                referrals = claims[
                    claims[col].notna() & 
                    (claims[col] != '') &
                    (claims[col] != claims['provider_npi'])
                ]
                
                for _, row in referrals.iterrows():
                    self.graph.add_edge(
                        row[col],
                        row['provider_npi'],
                        edge_type='REFERS_TO',
                        date=row['date_of_service']
                    )
    
    def _add_co_billing_edges(self, claims: pd.DataFrame):
        """Add CO_BILLS_WITH edges for providers billing same beneficiary in short window"""
        window = timedelta(days=self.settings.graph.co_billing_window_days)
        
        # Group claims by beneficiary
        ben_groups = claims.groupby('beneficiary_hash')
        
        for ben_hash, group in ben_groups:
            if len(group) < 2:
                continue
            
            providers = group['provider_npi'].unique()
            
            for i in range(len(providers)):
                for j in range(i+1, len(providers)):
                    # Check if claims fall within time window
                    prov_i_claims = group[group['provider_npi'] == providers[i]]
                    prov_j_claims = group[group['provider_npi'] == providers[j]]
                    
                    for _, claim_i in prov_i_claims.iterrows():
                        for _, claim_j in prov_j_claims.iterrows():
                            date_diff = abs(
                                claim_i['date_of_service'] - claim_j['date_of_service']
                            )
                            if date_diff <= window:
                                self.graph.add_edge(
                                    providers[i],
                                    providers[j],
                                    edge_type='CO_BILLS_WITH',
                                    beneficiary=ben_hash,
                                    date_diff_days=date_diff.days
                                )
                                break
    
    def _add_exclusion_nodes(self, exclusions: pd.DataFrame):
        """Add exclusion record nodes and FLAGGED_BY edges"""
        for _, row in exclusions.iterrows():
            exclusion_id = f"EXCL_{row['entity_npi']}_{row['exclusion_date'].strftime('%Y%m%d')}"
            
            self.graph.add_node(
                exclusion_id,
                node_type='Exclusion',
                source=row.get('exclusion_source', 'Unknown'),
                exclusion_date=row['exclusion_date'],
                reinstatement_date=row.get('reinstatement_date')
            )
            
            # Add edge from provider to exclusion record
            self.graph.add_edge(
                row['entity_npi'],
                exclusion_id,
                edge_type='FLAGGED_BY',
                date=row['exclusion_date']
            )
    
    def _add_ownership_edges(self, ownership: pd.DataFrame):
        """Add OWNED_BY edges and owner nodes"""
        for _, row in ownership.iterrows():
            owner_id = row['owner_hash']
            
            if owner_id not in self.graph:
                self.graph.add_node(
                    owner_id,
                    node_type='Owner',
                    role=row.get('owner_role', 'Unknown')
                )
            
            self.graph.add_edge(
                row['provider_npi'],
                owner_id,
                edge_type='OWNED_BY',
                ownership_percentage=row.get('ownership_percentage', 0)
            )
    
    def _add_facility_edges(self, claims: pd.DataFrame):
        """Add SHARES_FACILITY edges based on common service locations"""
        if 'facility_id' not in claims.columns and 'service_location' not in claims.columns:
            return
        
        facility_col = 'facility_id' if 'facility_id' in claims.columns else 'service_location'
        
        # Group claims by facility
        facility_groups = claims.groupby(facility_col)
        
        for facility, group in facility_groups:
            providers = group['provider_npi'].unique()
            
            if len(providers) > 1:
                # Add facility node
                facility_id = f"FAC_{facility}"
                self.graph.add_node(
                    facility_id,
                    node_type='Facility'
                )
                
                # Connect providers to facility
                for provider in providers:
                    self.graph.add_edge(
                        provider,
                        facility_id,
                        edge_type='SHARES_FACILITY'
                    )
    
    def _add_bank_account_edges(self, bank_accounts: pd.DataFrame):
        """Add SHARES_BANK_ACCOUNT edges"""
        # Group by hashed account number
        account_groups = bank_accounts.groupby('account_hash')
        
        for account_hash, group in account_groups:
            providers = group['provider_npi'].unique()
            
            if len(providers) > 1:
                # Add bank account node
                bank_id = f"BANK_{account_hash}"
                self.graph.add_node(
                    bank_id,
                    node_type='BankAccount'
                )
                
                # Connect providers to shared account
                for provider in providers:
                    self.graph.add_edge(
                        provider,
                        bank_id,
                        edge_type='SHARES_BANK_ACCOUNT'
                    )
    
    def _add_billing_similarity_edges(self, claims: pd.DataFrame):
        """Add SIMILAR_BILLING_PATTERN edges based on CPT code distribution"""
        # This requires procedure code data
        if 'procedure_code' not in claims.columns:
            return
        
        # Compute CPT distributions per provider
        provider_cpt = claims.groupby('provider_npi')['procedure_code'].apply(
            lambda x: x.value_counts(normalize=True)
        ).unstack(fill_value=0)
        
        providers = provider_cpt.index
        threshold = self.settings.graph.similarity_threshold
        
        for i in range(len(providers)):
            for j in range(i+1, len(providers)):
                similarity = 1 - cosine(
                    provider_cpt.loc[providers[i]], 
                    provider_cpt.loc[providers[j]]
                )
                
                if similarity >= threshold:
                    self.graph.add_edge(
                        providers[i],
                        providers[j],
                        edge_type='SIMILAR_BILLING_PATTERN',
                        similarity=similarity
                    )
    
    def get_graph_statistics(self) -> Dict:
        """Return graph statistics for validation"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {}
        }
        
        # Count node types
        for node, attr in self.graph.nodes(data=True):
            node_type = attr.get('node_type', 'Unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edge types
        for u, v, attr in self.graph.edges(data=True):
            edge_type = attr.get('edge_type', 'Unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        return stats