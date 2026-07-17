#!/usr/bin/env python3
"""
Graph-Based Provider Fraud Detection System
Main entry point for running the pipeline
"""

import argparse
import sys
from pathlib import Path
from fraud_detection.config.settings import Settings
from fraud_detection.pipeline.scoring import FraudDetectionPipeline

def main():
    parser = argparse.ArgumentParser(
        description='Graph-Based Provider Fraud Detection System'
    )
    
    parser.add_argument(
        '--provider-file',
        required=True,
        help='Path to provider data CSV'
    )
    
    parser.add_argument(
        '--claims-file',
        required=True,
        help='Path to claims data CSV'
    )
    
    parser.add_argument(
        '--exclusions-file',
        required=True,
        help='Path to exclusions data CSV'
    )
    
    parser.add_argument(
        '--ownership-file',
        help='Path to ownership disclosure CSV (optional)'
    )
    
    parser.add_argument(
        '--bank-file',
        help='Path to bank account data CSV (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Directory for output files'
    )
    
    parser.add_argument(
        '--investigator-capacity',
        type=int,
        default=50,
        help='Number of cases investigators can review'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Initialize settings
    settings = Settings()
    settings.pipeline.output_dir = args.output_dir
    settings.pipeline.investigator_capacity = args.investigator_capacity
    settings.pipeline.random_seed = args.seed
    
    # Initialize and run pipeline
    pipeline = FraudDetectionPipeline(settings)
    
    try:
        results = pipeline.run(
            provider_file=args.provider_file,
            claims_file=args.claims_file,
            exclusions_file=args.exclusions_file,
            ownership_file=args.ownership_file,
            bank_accounts_file=args.bank_file
        )
        
        # Print summary
        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Providers analyzed: {results['metadata']['num_providers']}")
        print(f"Claims processed: {results['metadata']['num_claims']}")
        print(f"Known exclusions: {results['metadata']['num_exclusions']}")
        print(f"\nModel Performance:")
        print(f"  AUC-PR: {results['evaluation']['auc_pr']:.4f}")
        print(f"  Precision@{args.investigator_capacity}: {results['evaluation']['precision_at_k']:.4f}")
        print(f"  Recall@{args.investigator_capacity}: {results['evaluation']['recall_at_k']:.4f}")
        
        if 'lift_at_k' in results['evaluation']:
            print(f"  Lift over baseline: {results['evaluation']['lift_at_k']:.2%}")
        
        print(f"\nTop risk factors:")
        for feature in results['explanations']['top_features'][:5]:
            print(f"  - {feature}")
        
        print(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error executing pipeline: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()