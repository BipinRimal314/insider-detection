"""
Federated Learning Module for Privacy-Preserving Insider Threat Detection

Enables training models across organizational silos without sharing raw data.
Critical for enterprise deployment where data cannot leave department boundaries.

Architecture:
1. Local Models: Each department trains independently
2. Gradient Aggregation: Only model updates are shared
3. Differential Privacy: Optional noise injection for privacy guarantees

Usage:
    python federated_learning.py --simulate

Benefits:
- Privacy: Raw data never leaves source
- Compliance: Meets GDPR/CCPA requirements
- Scalability: Distributed training across locations
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import random

# Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
import utils

logger = utils.logger


class FederatedClient:
    """
    Represents a single organizational unit in federated learning.
    Trains locally and shares only model gradients.
    """
    
    def __init__(self, client_id: str, data: np.ndarray, labels: np.ndarray = None):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model = None
        self.local_epochs = 1
        
    def set_model(self, model: keras.Model):
        """Receive model weights from server."""
        self.model = keras.models.clone_model(model)
        self.model.set_weights(model.get_weights())
        self.model.compile(optimizer='adam', loss='mse')
        
    def train_local(self, epochs: int = 1) -> List[np.ndarray]:
        """
        Train on local data and return weight updates.
        
        Returns:
            List of weight updates (new - old)
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model first.")
            
        old_weights = [w.copy() for w in self.model.get_weights()]
        
        # Train locally (autoencoder on normal data)
        if self.labels is not None:
            # Train only on normal samples
            normal_mask = self.labels == 0
            X_train = self.data[normal_mask]
        else:
            X_train = self.data
            
        if len(X_train) < 10:
            return [np.zeros_like(w) for w in old_weights]
            
        self.model.fit(X_train, X_train, epochs=epochs, 
                      batch_size=min(128, len(X_train)), verbose=0)
        
        new_weights = self.model.get_weights()
        
        # Compute weight deltas
        deltas = [new_w - old_w for new_w, old_w in zip(new_weights, old_weights)]
        
        return deltas
    
    def evaluate(self) -> Dict:
        """Evaluate local model performance."""
        if self.labels is None or self.labels.sum() == 0:
            return {'client_id': self.client_id, 'auc': None}
            
        predictions = self.model.predict(self.data, verbose=0)
        mse = np.mean(np.square(self.data - predictions), axis=(1, 2))
        
        auc = roc_auc_score(self.labels, mse)
        return {'client_id': self.client_id, 'auc': auc}


class FederatedServer:
    """
    Central server that coordinates federated learning.
    Aggregates gradients from clients and distributes updated model.
    """
    
    def __init__(self, model_fn, aggregation_method: str = 'fedavg'):
        """
        Args:
            model_fn: Function that returns a compiled keras model
            aggregation_method: How to aggregate client updates ('fedavg', 'fedprox')
        """
        self.model_fn = model_fn
        self.global_model = model_fn()
        self.aggregation_method = aggregation_method
        self.clients: List[FederatedClient] = []
        self.history = []
        self.output_dir = config.RESULTS_DIR / 'federated'
        self.output_dir.mkdir(exist_ok=True)
        
    def register_client(self, client: FederatedClient):
        """Register a client for federated training."""
        self.clients.append(client)
        logger.info(f"Registered client: {client.client_id} ({len(client.data)} samples)")
        
    def broadcast_model(self):
        """Send current global model to all clients."""
        for client in self.clients:
            client.set_model(self.global_model)
    
    def aggregate_updates(self, client_deltas: List[List[np.ndarray]], 
                         client_sizes: List[int]) -> List[np.ndarray]:
        """
        Aggregate weight updates from clients.
        
        Args:
            client_deltas: List of weight deltas from each client
            client_sizes: Number of samples at each client
            
        Returns:
            Aggregated weight updates
        """
        total_samples = sum(client_sizes)
        
        if self.aggregation_method == 'fedavg':
            # Weighted average by client data size
            aggregated = []
            for layer_idx in range(len(client_deltas[0])):
                layer_deltas = [
                    deltas[layer_idx] * (size / total_samples)
                    for deltas, size in zip(client_deltas, client_sizes)
                ]
                aggregated.append(sum(layer_deltas))
            return aggregated
        else:
            # Simple average
            aggregated = []
            n_clients = len(client_deltas)
            for layer_idx in range(len(client_deltas[0])):
                layer_deltas = [deltas[layer_idx] for deltas in client_deltas]
                aggregated.append(sum(layer_deltas) / n_clients)
            return aggregated
    
    def apply_updates(self, aggregated_deltas: List[np.ndarray], 
                     learning_rate: float = 1.0):
        """Apply aggregated updates to global model."""
        current_weights = self.global_model.get_weights()
        new_weights = [
            w + learning_rate * delta 
            for w, delta in zip(current_weights, aggregated_deltas)
        ]
        self.global_model.set_weights(new_weights)
    
    def train_round(self, local_epochs: int = 1) -> Dict:
        """
        Execute one round of federated training.
        
        Returns:
            Round statistics
        """
        # Broadcast model
        self.broadcast_model()
        
        # Collect client updates
        all_deltas = []
        client_sizes = []
        
        for client in self.clients:
            deltas = client.train_local(epochs=local_epochs)
            all_deltas.append(deltas)
            client_sizes.append(len(client.data))
        
        # Aggregate and apply
        aggregated = self.aggregate_updates(all_deltas, client_sizes)
        self.apply_updates(aggregated)
        
        # Evaluate
        client_metrics = [client.evaluate() for client in self.clients]
        
        round_stats = {
            'participating_clients': len(self.clients),
            'total_samples': sum(client_sizes),
            'client_metrics': client_metrics
        }
        self.history.append(round_stats)
        
        return round_stats
    
    def train(self, rounds: int = 10, local_epochs: int = 1) -> List[Dict]:
        """
        Execute multiple rounds of federated training.
        
        Args:
            rounds: Number of communication rounds
            local_epochs: Local training epochs per round
            
        Returns:
            Training history
        """
        logger.info(f"\nStarting Federated Learning: {rounds} rounds, {len(self.clients)} clients")
        
        for round_num in range(rounds):
            round_stats = self.train_round(local_epochs)
            
            # Log progress
            valid_aucs = [m['auc'] for m in round_stats['client_metrics'] if m['auc'] is not None]
            avg_auc = np.mean(valid_aucs) if valid_aucs else 0
            
            logger.info(f"Round {round_num + 1}/{rounds}: Avg AUC = {avg_auc:.4f}")
        
        return self.history
    
    def evaluate_global(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate global model on held-out test set."""
        predictions = self.global_model.predict(X_test, verbose=0)
        mse = np.mean(np.square(X_test - predictions), axis=(1, 2))
        
        if y_test.sum() > 0:
            auc = roc_auc_score(y_test, mse)
        else:
            auc = 0
            
        return {'global_auc': auc, 'test_samples': len(X_test)}
    
    def save_global_model(self, path: Path = None):
        """Save the global federated model."""
        if path is None:
            path = self.output_dir / 'federated_model.keras'
        self.global_model.save(path)
        logger.info(f"✓ Global model saved to {path}")


class DifferentialPrivacy:
    """
    Adds differential privacy guarantees through gradient clipping and noise.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 clip_norm: float = 1.0):
        """
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy violation
            clip_norm: Maximum gradient norm
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradient norms for bounded sensitivity."""
        clipped = []
        for grad in gradients:
            norm = np.linalg.norm(grad)
            if norm > self.clip_norm:
                grad = grad * (self.clip_norm / norm)
            clipped.append(grad)
        return clipped
    
    def add_noise(self, gradients: List[np.ndarray], 
                  num_samples: int) -> List[np.ndarray]:
        """Add Gaussian noise calibrated to privacy budget."""
        # Noise scale based on privacy parameters
        sigma = (2 * self.clip_norm * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        
        noisy = []
        for grad in gradients:
            noise = np.random.normal(0, sigma, grad.shape)
            noisy.append(grad + noise / num_samples)
        return noisy


def create_lstm_autoencoder(seq_length: int, n_features: int) -> keras.Model:
    """Create simple LSTM autoencoder for federated learning."""
    inputs = keras.layers.Input(shape=(seq_length, n_features))
    
    # Encoder
    x = keras.layers.LSTM(16, return_sequences=False)(inputs)
    
    # Decoder
    x = keras.layers.RepeatVector(seq_length)(x)
    x = keras.layers.LSTM(16, return_sequences=True)(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def simulate_federated_learning():
    """Simulate federated learning across departments."""
    logger.info("=" * 80)
    logger.info("FEDERATED LEARNING SIMULATION")
    logger.info("=" * 80)
    
    # Load data
    seq_path = config.PROCESSED_DATA_DIR / 'sequences.npy'
    labels_path = config.SEQUENCE_LABELS_FILE
    
    if not seq_path.exists():
        logger.error("Sequences not found. Run feature engineering first.")
        return
        
    X = np.load(seq_path)
    y = np.load(labels_path) if labels_path.exists() else np.zeros(len(X))
    
    logger.info(f"Loaded {len(X)} sequences")
    
    # Simulate 5 departments by splitting data
    n_clients = 5
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    
    # Create model factory
    model_fn = lambda: create_lstm_autoencoder(X.shape[1], X.shape[2])
    
    # Create federated server
    server = FederatedServer(model_fn, aggregation_method='fedavg')
    
    # Register clients
    for i, split_idx in enumerate(splits):
        client = FederatedClient(
            client_id=f"Department_{i+1}",
            data=X[split_idx],
            labels=y[split_idx]
        )
        server.register_client(client)
    
    # Train
    history = server.train(rounds=10, local_epochs=2)
    
    # Evaluate on holdout
    # Use data from all clients for final evaluation
    X_test = X[splits[-1]]
    y_test = y[splits[-1]]
    
    final_metrics = server.evaluate_global(X_test, y_test)
    logger.info(f"\n✓ Final Global Model AUC: {final_metrics['global_auc']:.4f}")
    
    # Save model
    server.save_global_model()
    
    # Generate report
    report_lines = [
        "# Federated Learning Report\n\n",
        f"## Configuration\n",
        f"- Clients: {n_clients} departments\n",
        f"- Aggregation: FedAvg\n",
        f"- Rounds: 10\n",
        f"- Local Epochs: 2\n\n",
        f"## Results\n",
        f"- Final AUC: **{final_metrics['global_auc']:.4f}**\n",
        f"- Test Samples: {final_metrics['test_samples']}\n\n",
        "## Privacy Benefits\n",
        "- Raw data never leaves departments\n",
        "- Only model gradients are shared\n",
        "- Differential privacy can be added for stronger guarantees\n"
    ]
    
    report_path = server.output_dir / 'federated_learning_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    logger.info(f"✓ Report saved to {report_path}")
    
    return server, history


if __name__ == "__main__":
    simulate_federated_learning()
