"""
Sensitivity Analysis Module
Tests the "Breaking Point" of the anomaly detection model by gradually reducing attack intensity.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config
import utils
from synthetic_attacks import AttackSimulator

logger = utils.logger

def run_sensitivity_analysis():
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS: FINDING THE BREAKING POINT")
    logger.info("=" * 60)
    
    simulator = AttackSimulator()
    baseline_df = simulator.load_baseline_data()
    
    if baseline_df is None:
        return

    # Define intensity levels (multipliers for the attack magnitude)
    # Testing micro-intensities to find breaking point since 0.01 was still 100% detected
    intensities = [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005]
    
    results = []
    
    # We focus on Data Exfiltration as the primary test case
    # Original: file_access_count: (50, 200) -> Mean ~125x baseline
    
    base_patterns = simulator.attack_patterns['data_exfiltration']['features'].copy()
    
    for intensity in intensities:
        logger.info(f"\nTesting Intensity: {intensity:.2f}")
        
        # Scale the attack patterns
        scaled_features = {}
        for feature, (min_val, max_val) in base_patterns.items():
            # Apply intensity scaling
            # We want to approach 1.0 (normal) as intensity -> 0
            # Formula: 1.0 + (Original_Multiplier - 1.0) * intensity
            
            new_min = 1.0 + (min_val - 1.0) * intensity
            new_max = 1.0 + (max_val - 1.0) * intensity
            scaled_features[feature] = (new_min, new_max)
            
        # Update simulator configuration
        simulator.attack_patterns['data_exfiltration']['features'] = scaled_features
        
        # Generate attacks
        # We perform multiple iterations to get a stable average
        n_iterations = 3
        detection_rates = []
        
        for i in range(n_iterations):
            # Generate ONLY data_exfiltration attacks for speed and clarity
            attacks_df = simulator.generate_attack('data_exfiltration', baseline_df)
            for k in range(9): # Generate 10 total instances per iteration
                 attacks_df = pd.concat([attacks_df, simulator.generate_attack('data_exfiltration', baseline_df)], ignore_index=True)
            
            # Add metadata columns required by inject_into_dataset
            attacks_df['attack_type'] = 'data_exfiltration'
            attacks_df['attack_instance'] = range(len(attacks_df))
            
            # CRITICAL FIX for Validity:
            # Enforce correlation between counts and Z-scores
            # In real data, if file_access_count is high, z-score MUST be high
            # We approximate this: Z = (Value - Mean) / Std
            stats = simulator.get_feature_statistics(baseline_df)
            
            for feature in ['file_access_count', 'email_count', 'after_hours_activity', 'daily_activity_count']:
                z_col = f"{feature}_self_zscore"
                if z_col in attacks_df.columns and feature in stats:
                    mu = stats[feature]['mean']
                    sigma = stats[feature]['std']
                    # Recalculate z-score based on the generated count value
                    attacks_df[z_col] = (attacks_df[feature] - mu) / (sigma + 1e-6)
            
            augmented_df = simulator.inject_into_dataset(baseline_df.copy(), attacks_df)
            eval_results = simulator.evaluate_model_on_attacks(augmented_df)
            
            if 'data_exfiltration' in eval_results:
                detection_rates.append(eval_results['data_exfiltration']['detection_rate'])
            else:
                detection_rates.append(0.0)
        
        avg_rate = np.mean(detection_rates)
        results.append({
            'Intensity': intensity,
            'Detection_Rate': avg_rate,
            'Avg_Multiplier': (scaled_features['file_access_count'][0] + scaled_features['file_access_count'][1]) / 2
        })
        
        logger.info(f"Intensity {intensity}: Detection Rate = {avg_rate:.1%}")

    # Save results
    df_results = pd.DataFrame(results)
    output_dir = config.RESULTS_DIR / 'sensitivity'
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'sensitivity_results.csv'
    df_results.to_csv(csv_path, index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Intensity'], df_results['Detection_Rate'], marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% Detection Threshold')
    
    plt.title('Model Sensitivity: Data Exfiltration vs Intensity', fontsize=14)
    plt.xlabel('Attack Intensity (Relative to Max)', fontsize=12)
    plt.ylabel('Detection Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for multipliers
    for i, row in df_results.iterrows():
        plt.annotate(f"{row['Avg_Multiplier']:.1f}x", 
                     (row['Intensity'], row['Detection_Rate']),
                     xytext=(0, 10), textcoords='offset points', ha='center')
    
    plot_path = output_dir / 'sensitivity_plot.png'
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Sensitivity plot saved to {plot_path}")
    
    # Generate Report
    report_path = output_dir / 'sensitivity_report.md'
    with open(report_path, 'w') as f:
        f.write("# Sensitivity Analysis Report\n\n")
        f.write("Determining the detection limits of the Isolation Forest model.\n\n")
        f.write("| Intensity | Avg Multiplier | Detection Rate |\n")
        f.write("|-----------|----------------|----------------|\n")
        for _, row in df_results.iterrows():
            f.write(f"| {row['Intensity']:.2f} | {row['Avg_Multiplier']:.1f}x | {row['Detection_Rate']:.1%} |\n")
        f.write("\n\n![Sensitivity Plot](sensitivity_plot.png)\n")
        
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
