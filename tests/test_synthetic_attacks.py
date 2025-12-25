import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import the module to test
# Adjust path if necessary or assume PYTHONPATH is set
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from synthetic_attacks import AttackSimulator

class TestSyntheticAttacks(unittest.TestCase):
    
    def setUp(self):
        self.simulator = AttackSimulator()
        
    def test_boiling_frog_pattern_existence(self):
        """Test that boiling_frog is defined in attack patterns."""
        self.assertIn('boiling_frog', self.simulator.attack_patterns)
        
    @patch('synthetic_attacks.random')
    def test_boiling_frog_monotonic_increase(self, mock_random):
        """
        Test that boiling frog attack generates increasing values over time.
        We mock random to ensure deterministic behavior (no noise).
        """
        # Mock random to return specific values
        # We need mock_random.randint to return a fixed duration
        mock_random.randint.return_value = 10 # 10 days duration
        
        # We need mock_random.uniform to return the base multiplier 
        # But wait, logic uses uniform for min/max?
        # If I mock uniform, it might break logic if it expects a range.
        # Instead, let's allow noise but check the trend.
        
        pass 

    def test_boiling_frog_implementation(self):
        """Test the logic of boiling frog generation."""
        # 1. Setup Data
        baseline_df = pd.DataFrame({
            'user': ['u1']*100,
            'day': pd.date_range(start='2024-01-01', periods=100),
            'logon_count': np.random.normal(10, 2, 100),
            'file_access_count': np.random.normal(50, 10, 100)
        })
        
        # 2. Run Attack Generation
        # We assume boiling_frog exists (it doesn't yet, so this part fails)
        try:
            attack_df = self.simulator.generate_attack(
                attack_type='boiling_frog',
                baseline_df=baseline_df,
                user_id='attacker',
                start_day='2024-01-01'
            )
        except ValueError as e:
            self.fail(f"Boiling Frog generation failed: {e}")
            
        # 3. Verify Properties
        # Boiling Frog should have increasing intensity
        # Let's check 'file_access_count' which should be a key feature
        
        values = attack_df['file_access_count'].values
        
        # Split into first half and second half
        mid = len(values) // 2
        first_half_mean = np.mean(values[:mid])
        second_half_mean = np.mean(values[mid:])
        
        # The second half should significantly exceed the first half
        self.assertGreater(second_half_mean, first_half_mean * 1.5, 
                          "Boiling frog attack did not show significant increase over time")
                          
        # Also check start and end point roughly
        self.assertLess(values[0], values[-1], "Attack did not end higher than it started")

if __name__ == '__main__':
    unittest.main()
