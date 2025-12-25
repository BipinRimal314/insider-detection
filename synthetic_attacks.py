"""
Synthetic Attack Injection Module

Generates controlled insider threat attack scenarios for:
1. Testing model sensitivity to different attack patterns
2. Augmenting training data with known attack signatures
3. Conducting controlled experiments with known ground truth

Attack Types Simulated:
- Data exfiltration (large file copies after hours)
- Credential abuse (access to unauthorized systems)
- Pre-departure behavior (job searching, mass copying before leaving)
- Privilege escalation (accessing systems above clearance)

Usage:
    python synthetic_attacks.py --inject

Output:
    data/processed/synthetic_attacks.csv
    data/processed/augmented_features.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

import config
import utils

logger = utils.logger


class AttackSimulator:
    """
    Generates synthetic insider threat attack patterns.
    """
    
    def __init__(self):
        self.output_dir = config.PROCESSED_DATA_DIR / 'synthetic'
        self.output_dir.mkdir(exist_ok=True)
        
        # Attack pattern definitions
        self.attack_patterns = {
            'data_exfiltration': {
                'description': 'Large file copies, USB usage, after-hours activity',
                'features': {
                    'file_access_count': (50, 200),  # min, max multiplier
                    'usb_connect_count': (5, 20),
                    'after_hours_activity': (20, 100),
                    'daily_activity_count': (2, 5)
                },
                'duration_days': (3, 14)
            },
            'credential_abuse': {
                'description': 'Accessing unusual systems, multiple failed logins',
                'features': {
                    'unique_pcs': (3, 10),
                    'logon_count': (5, 20),
                    'daily_activity_count': (3, 8)
                },
                'duration_days': (1, 7)
            },
            'pre_departure': {
                'description': 'Job searching, mass copying before leaving',
                'features': {
                    'web_visit_count': (5, 20),  # Job sites
                    'file_access_count': (10, 50),
                    'email_count': (3, 10),  # Sending to personal email
                    'after_hours_activity': (10, 50)
                },
                'duration_days': (14, 30)
            },
            'subtle_exfiltration': {
                'description': 'Low and slow data theft over extended period',
                'features': {
                    'file_access_count': (1.5, 3),  # Subtle increase
                    'email_count': (1.2, 2),
                    'after_hours_activity': (1.5, 3)
                },
                'duration_days': (30, 90)
            }
        }
    
    def load_baseline_data(self) -> Optional[pd.DataFrame]:
        """Load existing daily features to understand normal distribution."""
        daily_path = config.DAILY_FEATURES_FILE
        if not daily_path.exists():
            logger.error(f"Daily features not found at {daily_path}")
            return None
            
        df = pl.read_parquet(daily_path).to_pandas()
        logger.info(f"Loaded baseline data: {len(df)} samples")
        return df
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate mean and std for each feature."""
        exclude_cols = ['user', 'day', 'is_anomaly']
        # Select all numeric columns
        feature_cols = [c for c in df.columns if c not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df[c])]
        
        stats = {}
        for col in feature_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'p95': df[col].quantile(0.95),
                'p99': df[col].quantile(0.99)
            }
        return stats
    
    def generate_attack(self, attack_type: str, baseline_df: pd.DataFrame,
                       user_id: str = None, start_day: str = None) -> pd.DataFrame:
        """
        Generate synthetic attack data for a single attack scenario.
        
        Args:
            attack_type: Type of attack from attack_patterns
            baseline_df: DataFrame with normal user behavior
            user_id: User to simulate attack for (random if None)
            start_day: Start date of attack (random if None)
            
        Returns:
            DataFrame with synthetic attack features
        """
        if attack_type not in self.attack_patterns:
            raise ValueError(f"Unknown attack type: {attack_type}")
            
        pattern = self.attack_patterns[attack_type]
        
        # Get baseline statistics
        stats = self.get_feature_statistics(baseline_df)
        
        # Select or generate user
        if user_id is None:
            user_id = f"SYNTHETIC_{attack_type.upper()[:4]}_{random.randint(1000,9999)}"
        
        # Determine attack duration
        min_days, max_days = pattern['duration_days']
        duration = random.randint(min_days, max_days)
        
        # Generate start date
        if start_day is None:
            # Random date in the dataset range
            days = pd.to_datetime(baseline_df['day'].unique())
            start_idx = random.randint(0, len(days) - duration - 1)
            start_day = days[start_idx]
        else:
            start_day = pd.to_datetime(start_day)
        
        # Generate attack data
        attack_data = []
        
        for day_offset in range(duration):
            current_day = start_day + timedelta(days=day_offset)
            
            # Start with baseline normal values
            row = {'user': user_id, 'day': current_day, 'is_anomaly': 1}
            
            # Apply attack pattern modifications
            for feature, (min_mult, max_mult) in pattern['features'].items():
                if feature in stats:
                    base_value = stats[feature]['mean']
                    multiplier = random.uniform(min_mult, max_mult)
                    
                    # For subtle attacks, add noise
                    if attack_type == 'subtle_exfiltration':
                        noise = random.gauss(0, stats[feature]['std'] * 0.3)
                    else:
                        noise = random.gauss(0, stats[feature]['std'] * 0.1)
                    
                    row[feature] = max(0, base_value * multiplier + noise)
            
            # Fill remaining features with normal values
            for feature in stats:
                if feature not in row:
                    row[feature] = random.gauss(
                        stats[feature]['mean'], 
                        stats[feature]['std']
                    )
                    row[feature] = max(0, row[feature])
            
            attack_data.append(row)
        
        return pd.DataFrame(attack_data)
    
    def generate_attack_suite(self, baseline_df: pd.DataFrame, 
                             attacks_per_type: int = 3) -> pd.DataFrame:
        """
        Generate suite of synthetic attacks for all attack types.
        
        Args:
            baseline_df: Baseline normal data
            attacks_per_type: Number of attack instances per type
            
        Returns:
            DataFrame with all synthetic attacks
        """
        logger.info("=" * 60)
        logger.info("GENERATING SYNTHETIC ATTACK SUITE")
        logger.info("=" * 60)
        
        all_attacks = []
        attack_metadata = []
        
        for attack_type in self.attack_patterns:
            logger.info(f"\nGenerating {attacks_per_type} '{attack_type}' attacks...")
            
            for i in range(attacks_per_type):
                attack_df = self.generate_attack(attack_type, baseline_df)
                
                # Add attack metadata
                attack_df['attack_type'] = attack_type
                attack_df['attack_instance'] = i
                
                all_attacks.append(attack_df)
                
                attack_metadata.append({
                    'attack_type': attack_type,
                    'instance': i,
                    'user': attack_df['user'].iloc[0],
                    'start_day': attack_df['day'].min(),
                    'end_day': attack_df['day'].max(),
                    'duration_days': len(attack_df)
                })
        
        # Combine all attacks
        attacks_df = pd.concat(all_attacks, ignore_index=True)
        
        # Save metadata
        metadata_df = pd.DataFrame(attack_metadata)
        metadata_path = self.output_dir / 'attack_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        
        logger.info(f"\n✓ Generated {len(attacks_df)} synthetic attack records")
        logger.info(f"✓ Metadata saved to {metadata_path}")
        
        return attacks_df
    
    def inject_into_dataset(self, baseline_df: pd.DataFrame,
                           attacks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject synthetic attacks into baseline dataset.
        
        Args:
            baseline_df: Original dataset
            attacks_df: Synthetic attack data
            
        Returns:
            Combined dataset with attacks injected
        """
        logger.info("Injecting synthetic attacks into baseline data...")
        
        # Align columns
        common_cols = list(set(baseline_df.columns) & set(attacks_df.columns))
        
        # Mark original data as non-attack (if not already labeled)
        if 'is_anomaly' not in baseline_df.columns:
            baseline_df['is_anomaly'] = 0
        
        # Ensure attack_type column exists
        if 'attack_type' not in baseline_df.columns:
            baseline_df['attack_type'] = 'normal'
        
        attacks_df = attacks_df[common_cols + ['attack_type', 'attack_instance']]
        baseline_df['attack_instance'] = -1
        
        # Combine
        combined_df = pd.concat([
            baseline_df[common_cols + ['attack_type', 'attack_instance']], 
            attacks_df
        ], ignore_index=True)
        
        # Sort by user and day
        combined_df = combined_df.sort_values(['user', 'day']).reset_index(drop=True)
        
        logger.info(f"✓ Combined dataset: {len(combined_df)} records")
        logger.info(f"  - Normal: {len(baseline_df)}")
        logger.info(f"  - Attacks: {len(attacks_df)}")
        
        return combined_df
    
    def evaluate_model_on_attacks(self, augmented_df: pd.DataFrame) -> Dict:
        """
        Evaluate how well the model detects synthetic attacks.
        
        Args:
            augmented_df: Dataset with injected attacks
            
        Returns:
            Detection rates per attack type
        """
        logger.info("\nEvaluating model on synthetic attacks...")
        
        # Load Isolation Forest
        model_path = config.MODEL_PATHS.get('isolation_forest')
        if not model_path or not Path(model_path).exists():
            logger.warning("Model not found for evaluation")
            return {}
            
        import joblib
        model = joblib.load(model_path)
        
        # Prepare features
        exclude_cols = ['user', 'day', 'is_anomaly', 'attack_type', 'attack_instance']
        feature_cols = [c for c in augmented_df.columns if c not in exclude_cols
                       and pd.api.types.is_numeric_dtype(augmented_df[c])]
        
        X = augmented_df[feature_cols].fillna(0).values
        
        # Get predictions
        predictions = model.predict(X)
        predictions = (predictions == -1).astype(int)  # Convert to 0/1
        
        augmented_df['predicted_anomaly'] = predictions
        
        # Calculate detection rate per attack type
        results = {}
        for attack_type in self.attack_patterns:
            attack_mask = augmented_df['attack_type'] == attack_type
            if attack_mask.sum() > 0:
                detected = (augmented_df.loc[attack_mask, 'predicted_anomaly'] == 1).sum()
                total = attack_mask.sum()
                detection_rate = detected / total
                
                results[attack_type] = {
                    'total_records': int(total),
                    'detected': int(detected),
                    'detection_rate': detection_rate
                }
                logger.info(f"  {attack_type}: {detected}/{total} detected ({detection_rate:.1%})")
        
        # Overall
        normal_mask = augmented_df['attack_type'] == 'normal'
        fp = (augmented_df.loc[normal_mask, 'predicted_anomaly'] == 1).sum()
        results['false_positives'] = int(fp)
        results['false_positive_rate'] = fp / normal_mask.sum() if normal_mask.sum() > 0 else 0
        
        return results
    
    def run_full_pipeline(self, attacks_per_type: int = 3):
        """Run the full synthetic attack generation and evaluation pipeline."""
        # Load baseline
        baseline_df = self.load_baseline_data()
        if baseline_df is None:
            return
        
        # Generate attacks
        attacks_df = self.generate_attack_suite(baseline_df, attacks_per_type)
        
        # Save attacks
        attacks_path = self.output_dir / 'synthetic_attacks.csv'
        attacks_df.to_csv(attacks_path, index=False)
        logger.info(f"✓ Attacks saved to {attacks_path}")
        
        # Inject and evaluate
        augmented_df = self.inject_into_dataset(baseline_df, attacks_df)
        
        # Save augmented data
        augmented_path = self.output_dir / 'augmented_features.parquet'
        pl.from_pandas(augmented_df).write_parquet(augmented_path)
        logger.info(f"✓ Augmented data saved to {augmented_path}")
        
        # Evaluate
        results = self.evaluate_model_on_attacks(augmented_df)
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: Dict):
        """Generate synthetic attack evaluation report."""
        report_lines = [
            "# Synthetic Attack Evaluation Report\n\n",
            "## Attack Types\n"
        ]
        
        for attack_type, pattern in self.attack_patterns.items():
            report_lines.append(f"\n### {attack_type.replace('_', ' ').title()}\n")
            report_lines.append(f"{pattern['description']}\n")
            
            if attack_type in results:
                r = results[attack_type]
                report_lines.append(f"- Detection Rate: **{r['detection_rate']:.1%}**\n")
                report_lines.append(f"- Detected: {r['detected']}/{r['total_records']} records\n")
        
        report_lines.append("\n## False Positive Analysis\n")
        if 'false_positive_rate' in results:
            report_lines.append(f"- False Positives: {results['false_positives']}\n")
            report_lines.append(f"- FP Rate: {results['false_positive_rate']:.2%}\n")
        
        report_path = self.output_dir / 'attack_evaluation_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        logger.info(f"✓ Report saved to {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inject', action='store_true', help='Inject attacks and evaluate')
    parser.add_argument('--attacks-per-type', type=int, default=3, help='Number of attack instances per type')
    args = parser.parse_args()
    
    simulator = AttackSimulator()
    simulator.run_full_pipeline(attacks_per_type=args.attacks_per_type)
