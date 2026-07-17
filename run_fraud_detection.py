#!/usr/bin/env python3
"""
Complete executable script for fraud detection pipeline
Run this script to execute the full pipeline with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from utils.hashing import DataHasher
from utils.validation import DataValidator
from data.preprocessing import ClaimsPreprocessor, FeatureAggregator
from data.ingestion import DataIngestion
from graph.construction import GraphBuilder
from graph.features import GraphFeatureExtractor
from graph.embeddings import EmbeddingEnsemble
from models.baseline import BaselineEnsemble, TraditionalMLBaseline
from models.hybrid import HybridFraudDetector
from evaluation.metrics import FraudDetectionEvaluator
import networkx as nx

# Create output directories
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*60)
print("FRAUD DETECTION PIPELINE")
print("="*60)

# Step 1: Generate Sample Data
print("\n[1/8] Generating sample data...")

np.random.seed(42)

# Generate providers
n_providers = 200
n_fraud = 15  # Number of fraud providers

provider_data = []
for i in range(n_providers):
    provider_data.append({
        'npi': f'{np.random.randint(1000000000, 9999999999)}',
        'provider_name': f'Provider_{i}',
        'provider_type': np.random.choice(['Individual', 'Organization'], p=[0.8, 0.2]),
        'taxonomy_code': np.random.choice(['207Q00000X', '208D00000X', '363LF0000X', '207R00000X']),
        'specialty': np.random.choice(['Family Medicine', 'Internal Medicine', 'Nurse Practitioner', 'Radiology']),
        'license_state': np.random.choice(['NY', 'CA', 'TX', 'FL', 'PA', 'IL']),
        'enrollment_date': datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 730))
    })

providers = pd.DataFrame(provider_data)

# Designate fraud providers (first n_fraud providers)
fraud_providers = providers['npi'].iloc[:n_fraud].tolist()

# Generate claims
n_claims = 5000
claims_list = []

# Create fraud patients (small set shared among fraud ring)
fraud_patients = [f'PAT_FRAUD_{i:03d}' for i in range(20)]

# Create legitimate patients
legitimate_patients = [f'PAT_LEGIT_{i:04d}' for i in range(500)]

for i in range(n_claims):
    # 30% chance claim is from fraud ring
    if np.random.random() < 0.3:
        provider = np.random.choice(fraud_providers)
        patient = np.random.choice(fraud_patients)
        billed = np.random.uniform(500, 2000)  # Higher billing for fraud
    else:
        provider = np.random.choice(providers['npi'])
        patient = np.random.choice(legitimate_patients)
        billed = np.random.exponential(300)  # Normal billing
    
    claims_list.append({
        'claim_id': f'CLAIM_{i:06d}',
        'provider_npi': provider,
        'beneficiary_id': patient,
        'date_of_service': datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)),
        'billed_amount': billed,
        'paid_amount': billed * np.random.uniform(0.6, 0.95),
        'procedure_code': np.random.choice(['99213', '99214', '99215', '99203', '99204', '97110', '90837']),
        'diagnosis_code': np.random.choice(['I10', 'E11', 'J45', 'M54', 'F41']),
        'facility_id': np.random.choice(['FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005'])
    })

claims = pd.DataFrame(claims_list)

# Generate exclusions (fraud labels) - ensure same length as excluded_providers
excluded_providers = fraud_providers[:n_fraud]  # All fraud ring members are excluded

# Create exclusion dates
exclusion_dates = []
for i in range(n_fraud):
    exclusion_dates.append(datetime(2024, 1, 15) + timedelta(days=np.random.randint(0, 100)))

# Create exclusion sources
exclusion_sources = np.random.choice(['OIG', 'SAM.gov', 'State Medicaid'], n_fraud)

# Create exclusions DataFrame
exclusions = pd.DataFrame({
    'entity_npi': excluded_providers,
    'exclusion_date': exclusion_dates,
    'exclusion_source': exclusion_sources
})

print(f"  Exclusions created: {len(exclusions)} records")
print(f"  Excluded providers: {len(excluded_providers)}")
print(f"  Exclusion dates: {len(exclusion_dates)}")
print(f"  Exclusion sources: {len(exclusion_sources)}")

# Generate bank account data (shared accounts = fraud indicator)
# Create bank accounts for first 50 providers
bank_providers = providers['npi'].iloc[:50].tolist()
bank_accounts_list = []

for provider_npi in bank_providers:
    if provider_npi in fraud_providers:
        # Fraud ring shares one bank account
        account = 'BANK_FRAUD_RING_001'
    else:
        account = f'BANK_{np.random.randint(1000, 9999)}'
    
    bank_accounts_list.append({
        'provider_npi': provider_npi,
        'account_hash': account
    })

bank_accounts = pd.DataFrame(bank_accounts_list)

# Generate ownership data (optional)
ownership_list = []
for i in range(min(20, n_providers)):
    provider_npi = providers['npi'].iloc[i]
    if provider_npi in fraud_providers:
        # Fraud ring shares same owner
        owner_id = 'OWNER_FRAUD_RING'
    else:
        owner_id = f'OWNER_{np.random.randint(1000, 9999)}'
    
    ownership_list.append({
        'provider_npi': provider_npi,
        'owner_id': owner_id,
        'ownership_percentage': np.random.uniform(25, 100),
        'owner_role': np.random.choice(['Owner', 'Managing Member', 'Director'])
    })

ownership = pd.DataFrame(ownership_list)

# Save sample data
providers.to_csv('data/providers.csv', index=False)
claims.to_csv('data/claims.csv', index=False)
exclusions.to_csv('data/exclusions.csv', index=False)
bank_accounts.to_csv('data/bank_accounts.csv', index=False)
ownership.to_csv('data/ownership.csv', index=False)

print(f"\n  Generated data summary:")
print(f"    - {len(providers)} providers")
print(f"    - {len(claims)} claims")
print(f"    - {len(exclusions)} exclusion records")
print(f"    - {len(fraud_providers)} providers in fraud ring")
print(f"    - {len(bank_accounts)} bank account records")
print(f"    - {len(ownership)} ownership records")
print(f"  Fraud ring characteristics:")
print(f"    - Shared bank account: BANK_FRAUD_RING_001")
print(f"    - Shared patients: {len(fraud_patients)} patients")
print(f"    - Higher billing: $500-$2000 vs normal $0-$300")

# Step 2: Data Validation
print("\n[2/8] Validating data...")

validator = DataValidator()

# Validate providers
provider_valid, provider_errors = validator.validate_providers(providers)
print(f"  Providers valid: {provider_valid}")
if provider_errors:
    for error in provider_errors[:3]:  # Show first 3 errors
        print(f"    - {error}")

# Validate claims
claims_valid, claims_errors = validator.validate_claims(claims)
print(f"  Claims valid: {claims_valid}")
if claims_errors:
    for error in claims_errors[:3]:
        print(f"    - {error}")

# Validate exclusions
exclusions_valid, exclusions_errors = validator.validate_exclusions(exclusions)
print(f"  Exclusions valid: {exclusions_valid}")
if exclusions_errors:
    for error in exclusions_errors:
        print(f"    - {error}")

# Generate validation report
validation_report = validator.generate_validation_report(providers, claims, exclusions)
print(f"\n  Data statistics:")
for key, value in validation_report['statistics'].items():
    print(f"    - {key}: {value}")

# Step 3: Data Privacy (Hashing)
print("\n[3/8] Applying privacy hashing...")

hasher = DataHasher(salt="demo-salt-change-in-production")

# Hash sensitive fields
claims['beneficiary_hash'] = hasher.hash_series(claims['beneficiary_id'])
bank_accounts['account_hashed'] = hasher.hash_series(bank_accounts['account_hash'])
providers['provider_name_hash'] = hasher.hash_series(providers['provider_name'])
ownership['owner_hash'] = hasher.hash_series(ownership['owner_id'])

# Drop original sensitive fields
claims = claims.drop(columns=['beneficiary_id'])
bank_accounts = bank_accounts.drop(columns=['account_hash'])
providers = providers.drop(columns=['provider_name'])
ownership = ownership.drop(columns=['owner_id'])

# Rename hashed columns back
bank_accounts = bank_accounts.rename(columns={'account_hashed': 'account_hash'})

print(f"  ✓ Hashed beneficiary IDs, bank accounts, provider names, and owner IDs")

# Step 4: Graph Construction
print("\n[4/8] Building provider relationship graph...")

settings = Settings()
graph_builder = GraphBuilder(settings)

# Build the graph
try:
    graph = graph_builder.build_graph(
        providers=providers,
        claims=claims,
        exclusions=exclusions,
        ownership=ownership,
        bank_accounts=bank_accounts
    )
    
    stats = graph_builder.get_graph_statistics()
    print(f"  ✓ Graph constructed successfully")
    print(f"    - Nodes: {stats['num_nodes']}")
    print(f"    - Edges: {stats['num_edges']}")
    print(f"    - Node types: {dict(stats['node_types'])}")
    print(f"    - Edge types: {dict(stats['edge_types'])}")
    
except Exception as e:
    print(f"  ⚠️ Error building graph: {e}")
    print(f"  Building simplified graph...")
    # Build minimal graph
    graph = nx.MultiDiGraph()
    for _, row in providers.iterrows():
        graph.add_node(row['npi'], node_type='Provider')
    for _, row in claims.iterrows():
        graph.add_node(f"CLAIM_{row['claim_id']}", node_type='Claim')
        graph.add_edge(row['provider_npi'], f"CLAIM_{row['claim_id']}", edge_type='BILLS_FOR')

# Step 5: Feature Engineering
print("\n[5/8] Engineering features...")

# Extract graph features
feature_extractor = GraphFeatureExtractor(graph)
seed_providers = exclusions['entity_npi'].unique().tolist()

print(f"  Seed providers (known fraud): {len(seed_providers)}")

# Compute graph-structural features
print("  Computing centrality features...")
centrality_features = feature_extractor.compute_centrality_features()
print(f"    - Centrality features: {centrality_features.shape[1]} features")

# Compute PageRank baseline (guilt-by-association)
print("  Computing PageRank scores...")
pagerank_scores = feature_extractor.compute_pagerank_from_seeds(seed_providers)
print(f"    - PageRank scores computed for {len(pagerank_scores)} providers")

# Compute community features
try:
    print("  Computing community features...")
    community_features = feature_extractor.compute_community_features()
    print(f"    - Community features: {community_features.shape[1]} features")
except Exception as e:
    print(f"    - Community features: skipped ({str(e)[:100]})")
    community_features = pd.DataFrame()

# Compute local density
print("  Computing local density features...")
density_features = feature_extractor.compute_local_density_features()
print(f"    - Density features: {density_features.shape[1]} features")

# Compute distance to bad actors
print("  Computing distance to bad actors...")
distance_features = feature_extractor.compute_distance_to_bad_actors(seed_providers)
print(f"    - Distance features computed")

# Combine all graph features
graph_features_list = [centrality_features, pagerank_scores]

if not community_features.empty:
    graph_features_list.append(community_features)

graph_features_list.extend([density_features, distance_features])

graph_features = pd.concat(graph_features_list, axis=1)
graph_features = graph_features.fillna(0)
graph_features = graph_features.replace([np.inf, -np.inf], 0)

print(f"  ✓ Total graph-structural features: {graph_features.shape[1]}")

# Extract claims-behavioral features
print("  Computing claims-behavioral features...")
preprocessor = ClaimsPreprocessor()
claims_clean = preprocessor.clean_claims_data(claims)

temporal_features = preprocessor.extract_temporal_features(claims_clean)
billing_features = preprocessor.extract_billing_features(claims_clean)
upcoding_features = preprocessor.detect_upcoding_patterns(claims_clean)
impossible_days = preprocessor.detect_impossible_days(claims_clean)

# Merge all claims features
claims_features = providers[['npi']].rename(columns={'npi': 'provider_npi'})
claims_features = claims_features.set_index('provider_npi')

for feat_name, feat_df in [
    ('temporal', temporal_features),
    ('billing', billing_features),
    ('upcoding', upcoding_features),
    ('impossible_days', impossible_days)
]:
    if not feat_df.empty and 'provider_npi' in feat_df.columns:
        feat_df = feat_df.set_index('provider_npi')
        claims_features = claims_features.join(feat_df, how='left')

claims_features = claims_features.fillna(0)
claims_features = claims_features.replace([np.inf, -np.inf], 0)

print(f"  ✓ Claims-behavioral features: {claims_features.shape[1]} features")

# Step 5b: Compute Graph Embeddings (if graph is large enough)
print("\n[5b/8] Computing graph embeddings...")

try:
    # Get the provider subgraph for embeddings
    G_providers = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if u in providers['npi'].values and v in providers['npi'].values:
            if G_providers.has_edge(u, v):
                G_providers[u][v]['weight'] = G_providers[u][v].get('weight', 0) + 1
            else:
                G_providers.add_edge(u, v, weight=1)
    
    if G_providers.number_of_nodes() > 10:
        embedding_ensemble = EmbeddingEnsemble(
            methods=['node2vec', 'spectral', 'structural']
        )
        graph_embeddings = embedding_ensemble.fit(G_providers)
        print(f"  ✓ Generated {graph_embeddings.shape[1]} graph embedding features")
        
        # Add embeddings to features
        graph_features_with_emb = graph_features.join(
            graph_embeddings, 
            how='left'
        ).fillna(0)
    else:
        print("  ⚠️ Not enough provider nodes for embeddings, skipping")
        graph_features_with_emb = graph_features.copy()
        graph_embeddings = pd.DataFrame()
        
except Exception as e:
    print(f"  ⚠️ Embedding computation failed: {e}")
    print(f"  Continuing with standard features only")
    graph_features_with_emb = graph_features.copy()
    graph_embeddings = pd.DataFrame()

# Step 5c: Compute Baseline Models for Comparison
print("\n[5c/8] Computing baseline models for comparison...")

# Graph-based baselines
print("  Computing graph-based baselines...")
baseline_ensemble = BaselineEnsemble()

try:
    # Create undirected graph for baselines
    G_undirected = nx.Graph()
    for u, v in graph.edges():
        G_undirected.add_edge(u, v)
    
    baseline_scores_df = baseline_ensemble.fit(G_undirected, seed_providers)
    print(f"  ✓ Baseline methods computed: {list(baseline_scores_df.columns)}")
    
except Exception as e:
    print(f"  ⚠️ Baseline computation failed: {e}")
    # Create dummy baseline scores
    dummy_scores = pd.Series(0.0, index=providers['npi'])
    baseline_scores_df = pd.DataFrame({
        'pagerank': dummy_scores,
        'rwr': dummy_scores,
        'label_propagation': dummy_scores,
        'ensemble_score': dummy_scores
    })

# Traditional ML baseline (no graph features)
print("  Training traditional ML baseline (claims features only)...")

# Prepare labels
labels = pd.Series(0, index=claims_features.index)
labels[labels.index.isin(seed_providers)] = 1

# Align features and labels
common_idx = claims_features.index.intersection(labels.index)
X_traditional = claims_features.loc[common_idx]
y_traditional = labels.loc[common_idx]

# Train traditional baseline
if len(X_traditional) > 10 and y_traditional.sum() > 0:
    traditional_baseline = TraditionalMLBaseline(model_type='random_forest')
    traditional_baseline.fit(X_traditional, y_traditional)
    traditional_scores = traditional_baseline.predict_risk(X_traditional)
    print(f"  ✓ Traditional baseline trained on {X_traditional.shape[1]} claims features")
else:
    print(f"  ⚠️ Not enough data for traditional baseline")
    traditional_scores = np.zeros(len(common_idx))

# Step 6: Model Training with Multiple Variants
print("\n[6/8] Training hybrid fraud detection models...")

# Prepare labels for all models
labels = pd.Series(0, index=graph_features_with_emb.index)
labels[labels.index.isin(seed_providers)] = 1

print(f"  Labels: {labels.sum()} positive, {len(labels) - labels.sum()} negative")

# Combine all features (graph + claims + embeddings)
all_features = graph_features_with_emb.join(claims_features, how='outer')
all_features = all_features.fillna(0)
all_features = all_features.replace([np.inf, -np.inf], 0)

# Align features and labels
common_idx = all_features.index.intersection(labels.index)
X_full = all_features.loc[common_idx]
y_full = labels.loc[common_idx]

print(f"  Full feature set: {X_full.shape[1]} features for {len(X_full)} providers")

# Train different model variants
models_trained = {}

# 1. Full model (all features)
if len(X_full) > 20 and y_full.sum() >= 2:
    print("  Training Full Hybrid Model...")
    hybrid_detector_full = HybridFraudDetector(random_state=42)
    hybrid_detector_full.train(X_full, y_full, validation_size=0.2)
    models_trained['full'] = hybrid_detector_full
    risk_scores_full = hybrid_detector_full.predict_risk_scores(X_full)
else:
    print("  ⚠️ Not enough data for full model")
    risk_scores_full = np.zeros(len(common_idx))

# 2. Graph-only model
X_graph_only = graph_features_with_emb.loc[common_idx]
if len(X_graph_only) > 20 and y_full.sum() >= 2:
    print("  Training Graph-Only Model...")
    hybrid_detector_graph = HybridFraudDetector(random_state=42)
    hybrid_detector_graph.train(X_graph_only, y_full, validation_size=0.2)
    models_trained['graph'] = hybrid_detector_graph
    risk_scores_graph = hybrid_detector_graph.predict_risk_scores(X_graph_only)
else:
    risk_scores_graph = np.zeros(len(common_idx))

# 3. Baseline-enhanced model (claims + simple graph baselines)
X_with_baseline = claims_features.loc[common_idx].copy()
for col in baseline_scores_df.columns:
    X_with_baseline[col] = baseline_scores_df[col].reindex(common_idx).fillna(0).values

if len(X_with_baseline) > 20 and y_full.sum() >= 2:
    print("  Training Baseline-Enhanced Model...")
    hybrid_detector_baseline = HybridFraudDetector(random_state=42)
    hybrid_detector_baseline.train(X_with_baseline, y_full, validation_size=0.2)
    models_trained['baseline'] = hybrid_detector_baseline
    risk_scores_baseline = hybrid_detector_baseline.predict_risk_scores(X_with_baseline)
else:
    risk_scores_baseline = np.zeros(len(common_idx))

print(f"  ✓ Trained {len(models_trained)} model variants")

# Step 7: Generate Risk Scores and Compare Models
print("\n[7/8] Generating risk scores and comparing models...")

# Get traditional ML scores
traditional_scores_aligned = pd.Series(traditional_scores, index=X_traditional.index)
traditional_scores_full = traditional_scores_aligned.reindex(common_idx).fillna(0).values

# Get pure PageRank baseline scores
pagerank_aligned = baseline_scores_df['pagerank'].reindex(common_idx).fillna(0).values

# Create comprehensive results DataFrame
results_df = pd.DataFrame({
    'provider_npi': common_idx,
    'risk_score_full': risk_scores_full,
    'risk_score_graph_only': risk_scores_graph,
    'risk_score_baseline_enhanced': risk_scores_baseline,
    'risk_score_traditional_ml': traditional_scores_full,
    'pagerank_baseline': pagerank_aligned,
    'is_fraud': y_full.values
})

# Sort by full model risk score
results_df = results_df.sort_values('risk_score_full', ascending=False)

print(f"\n  Top 10 highest-risk providers (Full Model):")
for idx, row in results_df.head(10).iterrows():
    fraud_status = "⚠️ FRAUD" if row['is_fraud'] else "✓ Clean"
    print(f"    {row['provider_npi']}: Full={row['risk_score_full']:.4f}, "
          f"Trad={row['risk_score_traditional_ml']:.4f}, "
          f"Status: {fraud_status}")

# Generate explanations from full model if available
if 'full' in models_trained:
    try:
        explanations_full = models_trained['full'].explain_predictions(X_full, top_k=10)
        
        print(f"\n  Top predictive features (Full Model):")
        for feature in explanations_full['top_features'][:10]:
            importance = explanations_full['feature_importance'].get(feature, 0)
            # Classify feature type
            if any(feature.startswith(prefix) for prefix in ['node2vec', 'spectral', 'structural']):
                feat_type = "[Embedding]"
            elif any(feature.startswith(prefix) for prefix in ['pagerank', 'rwr', 'label_prop']):
                feat_type = "[Baseline]"
            elif feature in claims_features.columns:
                feat_type = "[Claims]"
            else:
                feat_type = "[Graph]"
            print(f"    {feat_type} {feature}: {importance:.4f}")
    except Exception as e:
        print(f"  ⚠️ Could not generate explanations: {e}")

# Step 8: Comprehensive Model Evaluation
print("\n[8/8] Evaluating and comparing all models...")

evaluator = FraudDetectionEvaluator(investigator_capacity=20)

# Evaluate each model
models = {
    'Full Hybrid Model': (y_full.values, risk_scores_full),
    'Graph-Only Model': (y_full.values, risk_scores_graph),
    'Baseline-Enhanced': (y_full.values, risk_scores_baseline),
    'Traditional ML': (y_full.values, traditional_scores_full),
    'Pure PageRank': (y_full.values, pagerank_aligned)
}

evaluation_results = {}
for model_name, (y_true, y_scores) in models.items():
    try:
        evaluation_results[model_name] = evaluator.evaluate(
            y_true, 
            y_scores,
            baseline_scores=pagerank_aligned if model_name != 'Pure PageRank' else None
        )
    except Exception as e:
        print(f"  ⚠️ Could not evaluate {model_name}: {e}")
        evaluation_results[model_name] = {
            'auc_pr': 0.0,
            'precision_at_k': 0.0,
            'recall_at_k': 0.0
        }

# Print comparison table
print(f"\n{'='*80}")
print("MODEL COMPARISON")
print(f"{'='*80}")
print(f"{'Model':<25} {'AUC-PR':<10} {'Precision@20':<15} {'Recall@20':<12}")
print("-"*80)

for model_name, metrics in evaluation_results.items():
    auc_pr = metrics.get('auc_pr', 0)
    prec_k = metrics.get('precision_at_k', 0)
    rec_k = metrics.get('recall_at_k', 0)
    
    print(f"{model_name:<25} {auc_pr:<10.4f} {prec_k:<15.4f} {rec_k:<12.4f}")

# Find best model
valid_results = {k: v for k, v in evaluation_results.items() if v.get('auc_pr', 0) > 0}
if valid_results:
    best_model = max(valid_results.items(), key=lambda x: x[1].get('auc_pr', 0))
    print(f"\n✓ Best performing model: {best_model[0]} (AUC-PR: {best_model[1]['auc_pr']:.4f})")
else:
    best_model = ('Full Hybrid Model', {'auc_pr': 0.0})
    print(f"\n⚠️ No model achieved positive AUC-PR")

# Generate detailed report for full model
top_20_report = evaluator.generate_report(
    common_idx.tolist(),
    y_full.values,
    risk_scores_full,
    top_k=20
)

print(f"\n{'='*80}")
print("TOP 20 HIGHEST RISK PROVIDERS")
print(f"{'='*80}")
print(f"{'Rank':<6} {'Provider NPI':<15} {'Risk Score':<12} {'Fraud?':<8} {'Cum.Precision':<15}")
print("-"*80)
for _, row in top_20_report.head(20).iterrows():
    fraud_mark = "⚠️ YES" if row['is_fraud'] else "No"
    print(f"{row['rank']:<6} {row['provider_id']:<15} {row['risk_score']:<12.4f} "
          f"{fraud_mark:<8} {row['cumulative_precision']:<15.4f}")

# Save all results
print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

# Save comprehensive risk scores
results_df.to_csv('output/all_model_risk_scores.csv', index=False)
print("  ✓ All model risk scores saved to output/all_model_risk_scores.csv")

# Save model comparison
model_comparison_data = {}
for model_name, metrics in evaluation_results.items():
    model_comparison_data[model_name] = {
        'AUC-PR': metrics.get('auc_pr', 0),
        'Precision@20': metrics.get('precision_at_k', 0),
        'Recall@20': metrics.get('recall_at_k', 0),
        'Optimal_F1': metrics.get('optimal_f1', 0)
    }

model_comparison = pd.DataFrame(model_comparison_data).T
model_comparison.to_csv('output/model_comparison.csv')
print("  ✓ Model comparison saved to output/model_comparison.csv")

# Save evaluation metrics
with open('output/evaluation_all_models.json', 'w') as f:
    eval_serializable = {}
    for model_name, metrics in evaluation_results.items():
        eval_serializable[model_name] = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                eval_serializable[model_name][k] = v.tolist()
            elif isinstance(v, dict):
                eval_serializable[model_name][k] = {
                    kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                    for kk, vv in v.items()
                }
            else:
                eval_serializable[model_name][k] = v
    json.dump(eval_serializable, f, indent=2)
print("  ✓ Comprehensive evaluation saved to output/evaluation_all_models.json")

# Save feature importance if available
if 'full' in models_trained:
    try:
        with open('output/feature_importance.json', 'w') as f:
            json.dump({
                'feature_importance': explanations_full['feature_importance'],
                'top_features': explanations_full['top_features']
            }, f, indent=2)
        print("  ✓ Feature importance saved to output/feature_importance.json")
    except:
        pass

# Save best model
if 'full' in models_trained:
    try:
        models_trained['full'].save_model('models/best_hybrid_model.pkl')
        print("  ✓ Best model saved to models/best_hybrid_model.pkl")
    except Exception as e:
        print(f"  ⚠️ Could not save model: {e}")

# Generate final summary report
summary = {
    'pipeline_version': '1.0.0',
    'execution_timestamp': datetime.now().isoformat(),
    'data_summary': {
        'total_providers': len(providers),
        'total_claims': len(claims),
        'known_fraud_cases': len(exclusions),
        'fraud_ring_size': len(fraud_providers)
    },
    'feature_summary': {
        'graph_features': graph_features.shape[1],
        'embedding_features': graph_embeddings.shape[1] if not graph_embeddings.empty else 0,
        'claims_features': claims_features.shape[1],
        'total_features': X_full.shape[1],
        'baseline_methods': list(baseline_scores_df.columns)
    },
    'model_comparison': model_comparison.to_dict(),
    'best_model': {
        'name': best_model[0],
        'auc_pr': best_model[1].get('auc_pr', 0),
        'precision_at_k': best_model[1].get('precision_at_k', 0)
    },
    'top_risk_providers': [
        {
            'npi': row['provider_id'],
            'risk_score': float(row['risk_score']),
            'rank': int(row['rank']),
            'is_fraud': bool(row['is_fraud'])
        }
        for _, row in top_20_report.head(10).iterrows()
    ]
}

with open('output/pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  ✓ Pipeline summary saved to output/pipeline_summary.json")

print(f"\n{'='*80}")
print("PIPELINE EXECUTION COMPLETE")
print(f"{'='*80}")
print(f"\nKey Findings:")
print(f"  1. Best model: {best_model[0]}")
print(f"  2. AUC-PR: {best_model[1].get('auc_pr', 0):.4f}")
print(f"  3. Precision@20: {best_model[1].get('precision_at_k', 0):.4f}")

if 'Traditional ML' in evaluation_results and 'Full Hybrid Model' in evaluation_results:
    trad_auc = evaluation_results['Traditional ML'].get('auc_pr', 0.001)
    full_auc = evaluation_results['Full Hybrid Model'].get('auc_pr', 0)
    if trad_auc > 0:
        improvement = (full_auc / trad_auc - 1) * 100
        print(f"  4. Graph features improved detection by {improvement:.1f}%")

print(f"\nNext steps:")
print(f"  1. Review model comparison in output/model_comparison.csv")
print(f"  2. Analyze top-risk providers in output/all_model_risk_scores.csv")
print(f"  3. Investigate why graph features improved/worsened detection")
print(f"  4. Use best model for production scoring")
print(f"  5. Check output/pipeline_summary.json for complete results")