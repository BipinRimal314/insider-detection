import unittest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hyperparameter_sensitivity import HyperparameterSensitivity

class TestHyperparameterSensitivity(unittest.TestCase):
    def setUp(self):
        self.analyzer = HyperparameterSensitivity()

    def test_initialization(self):
        """Test that the class initializes with correct default paths and configs."""
        self.assertTrue(hasattr(self.analyzer, 'results_dir'))
        self.assertTrue(hasattr(self.analyzer, 'config'))
        self.assertTrue((self.analyzer.results_dir / 'sensitivity').exists() or 
                        (self.analyzer.results_dir).exists(), "Results directory should exist")

    def test_lstm_config_structure(self):
        """Test that LSTM configuration ranges are defined."""
        self.assertTrue(hasattr(self.analyzer, 'lstm_param_grid'))
        self.assertIn('units', self.analyzer.lstm_param_grid)
        self.assertIn('dropout', self.analyzer.lstm_param_grid)

    def test_isolation_forest_config_structure(self):
        """Test that Isolation Forest configuration ranges are defined."""
        self.assertTrue(hasattr(self.analyzer, 'if_param_grid'))
        self.assertIn('n_estimators', self.analyzer.if_param_grid)

if __name__ == '__main__':
    unittest.main()
