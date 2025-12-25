import unittest
import os
import sys
import numpy as np
import pandas as pd
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

    @patch('hyperparameter_sensitivity.utils')
    @patch('hyperparameter_sensitivity.LSTMAutoencoder')
    def test_run_lstm_sensitivity(self, MockLSTM, mock_utils):
        """Test the execution of LSTM sensitivity analysis."""
        # Setup mocks
        mock_model_instance = MockLSTM.return_value
        # Mock train_and_evaluate to return dummy history and auc
        mock_model_instance.train_and_evaluate.return_value = (
            MagicMock(history={'loss': [0.1], 'val_loss': [0.15]}),  # history
            0.95,  # auc
            0.05,  # threshold
            None,  # scaler
            100    # X_val length
        )
        
        # Mock utils data loading
        mock_utils.load_and_preprocess_data.return_value = (
            np.zeros((100, 10, 5)), # X_train
            np.zeros((100, 10, 5)), # X_val
            np.zeros((100, 10, 5)), # X_test
            np.zeros(100),          # y_test
            None                    # scaler
        )
        
        # Limit grid for testing speed
        self.analyzer.lstm_param_grid['units'] = [[16, 8]]
        self.analyzer.lstm_param_grid['dropout'] = [0.0]
        
        results = self.analyzer.run_lstm_sensitivity()
        
        # Verify results structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self.assertIn('units', results.columns)
        self.assertIn('dropout', results.columns)
        self.assertIn('auc', results.columns)
        
        # Verify logic called model training
        MockLSTM.assert_called()
        mock_model_instance.train_and_evaluate.assert_called()
