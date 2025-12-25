"""
Real-Time Streaming Detection Module

Enables continuous monitoring and detection of insider threats.
Bridges research to production with operational deployment capabilities.

Architecture:
    Log Sources → Feature Buffer → Online Scoring → Alert Queue → Dashboard

Features:
1. Sliding window feature computation
2. Online model inference
3. Alert prioritization and deduplication
4. REST API for integration

Usage:
    python realtime_detection.py --serve --port 8080

For production, pair with:
- Apache Kafka for log ingestion
- Redis for feature caching
- Grafana for visualization
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from datetime import datetime, timedelta
import json
import threading
import time

# Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib

import config
import utils

logger = utils.logger


class FeatureBuffer:
    """
    Maintains sliding window of features for real-time computation.
    """
    
    def __init__(self, window_size: int = 30, feature_names: List[str] = None):
        self.window_size = window_size
        self.feature_names = feature_names or []
        self.user_buffers: Dict[str, Deque] = {}
        self.lock = threading.Lock()
        
    def add_event(self, user_id: str, event: Dict):
        """Add an event to user's feature buffer."""
        with self.lock:
            if user_id not in self.user_buffers:
                self.user_buffers[user_id] = deque(maxlen=self.window_size)
            
            # Extract features from event
            features = self._extract_features(event)
            self.user_buffers[user_id].append({
                'timestamp': event.get('timestamp', datetime.now()),
                'features': features
            })
    
    def _extract_features(self, event: Dict) -> Dict:
        """Extract relevant features from a raw event."""
        return {
            'activity_type': event.get('activity_type', 'unknown'),
            'hour': datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat())).hour,
            'is_after_hours': 1 if (event.get('hour', 12) < 7 or event.get('hour', 12) > 19) else 0
        }
    
    def get_user_features(self, user_id: str) -> Optional[np.ndarray]:
        """Get aggregated features for a user."""
        with self.lock:
            if user_id not in self.user_buffers or len(self.user_buffers[user_id]) == 0:
                return None
            
            buffer = self.user_buffers[user_id]
            
            # Aggregate features
            activity_count = len(buffer)
            after_hours_count = sum(1 for e in buffer if e['features'].get('is_after_hours', 0))
            
            # Create feature vector
            features = np.array([
                activity_count,
                after_hours_count,
                after_hours_count / max(activity_count, 1)
            ])
            
            return features
    
    def get_all_user_features(self) -> Dict[str, np.ndarray]:
        """Get features for all users."""
        with self.lock:
            return {
                user_id: self.get_user_features(user_id)
                for user_id in self.user_buffers
            }


class OnlineScorer:
    """
    Real-time scoring using pre-trained models.
    """
    
    def __init__(self):
        self.models = {}
        self.thresholds = {}
        self.scaler = None
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models for inference."""
        # Isolation Forest
        if_path = config.MODEL_PATHS.get('isolation_forest')
        if if_path and Path(if_path).exists():
            self.models['isolation_forest'] = joblib.load(if_path)
            logger.info("✓ Loaded Isolation Forest")
            
        # Scaler
        scaler_path = config.MODEL_PATHS.get('static_scaler')
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            
    def score(self, features: np.ndarray, model_name: str = 'isolation_forest') -> float:
        """
        Score a feature vector using specified model.
        
        Args:
            features: 1D or 2D feature array
            model_name: Which model to use
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        if model_name not in self.models:
            return 0.0
            
        model = self.models[model_name]
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale if scaler available
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except:
                pass  # Feature mismatch, use raw
        
        # Score
        if hasattr(model, 'decision_function'):
            # Isolation Forest: lower = more anomalous, so negate
            return -float(model.decision_function(features)[0])
        elif hasattr(model, 'score_samples'):
            return -float(model.score_samples(features)[0])
        else:
            return 0.0
    
    def score_batch(self, user_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Score all users in batch."""
        scores = {}
        for user_id, features in user_features.items():
            if features is not None:
                scores[user_id] = self.score(features)
        return scores


class AlertManager:
    """
    Manages alert generation, prioritization, and deduplication.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.alerts: List[Dict] = []
        self.alert_history: Dict[str, datetime] = {}  # For deduplication
        self.dedup_window = timedelta(hours=24)
        self.lock = threading.Lock()
        
    def process_scores(self, scores: Dict[str, float]) -> List[Dict]:
        """
        Process scores and generate alerts for high-risk users.
        
        Args:
            scores: User ID to score mapping
            
        Returns:
            List of new alerts
        """
        new_alerts = []
        current_time = datetime.now()
        
        with self.lock:
            for user_id, score in scores.items():
                if score > self.threshold:
                    # Check deduplication
                    last_alert = self.alert_history.get(user_id)
                    if last_alert and (current_time - last_alert) < self.dedup_window:
                        continue  # Skip duplicate
                    
                    # Create alert
                    severity = self._calculate_severity(score)
                    alert = {
                        'timestamp': current_time.isoformat(),
                        'user_id': user_id,
                        'score': score,
                        'severity': severity,
                        'status': 'new'
                    }
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    self.alert_history[user_id] = current_time
            
        return new_alerts
    
    def _calculate_severity(self, score: float) -> str:
        """Map score to severity level."""
        if score > 0.95:
            return 'critical'
        elif score > 0.85:
            return 'high'
        elif score > 0.75:
            return 'medium'
        else:
            return 'low'
    
    def get_alerts(self, status: str = None, limit: int = 100) -> List[Dict]:
        """Get alerts, optionally filtered by status."""
        with self.lock:
            alerts = self.alerts
            if status:
                alerts = [a for a in alerts if a['status'] == status]
            return alerts[-limit:]
    
    def update_alert_status(self, user_id: str, new_status: str):
        """Update an alert's status."""
        with self.lock:
            for alert in self.alerts:
                if alert['user_id'] == user_id and alert['status'] == 'new':
                    alert['status'] = new_status
                    break


class RealTimeDetector:
    """
    Main real-time detection orchestrator.
    """
    
    def __init__(self):
        self.feature_buffer = FeatureBuffer(window_size=30)
        self.scorer = OnlineScorer()
        self.alert_manager = AlertManager(threshold=0.7)
        self.running = False
        self.output_dir = config.RESULTS_DIR / 'realtime'
        self.output_dir.mkdir(exist_ok=True)
        
    def process_event(self, event: Dict) -> Optional[Dict]:
        """
        Process a single event and potentially generate an alert.
        
        Args:
            event: Raw log event with user_id, timestamp, activity_type
            
        Returns:
            Alert dict if generated, None otherwise
        """
        user_id = event.get('user_id')
        if not user_id:
            return None
        
        # Add to buffer
        self.feature_buffer.add_event(user_id, event)
        
        # Score user
        features = self.feature_buffer.get_user_features(user_id)
        if features is None:
            return None
            
        score = self.scorer.score(features)
        
        # Check for alert
        alerts = self.alert_manager.process_scores({user_id: score})
        return alerts[0] if alerts else None
    
    def process_batch(self, events: List[Dict]) -> List[Dict]:
        """Process a batch of events."""
        for event in events:
            user_id = event.get('user_id')
            if user_id:
                self.feature_buffer.add_event(user_id, event)
        
        # Score all users
        user_features = self.feature_buffer.get_all_user_features()
        scores = self.scorer.score_batch(user_features)
        
        # Generate alerts
        return self.alert_manager.process_scores(scores)
    
    def get_dashboard_data(self) -> Dict:
        """Get current state for dashboard."""
        return {
            'timestamp': datetime.now().isoformat(),
            'monitored_users': len(self.feature_buffer.user_buffers),
            'total_alerts': len(self.alert_manager.alerts),
            'new_alerts': len(self.alert_manager.get_alerts(status='new')),
            'recent_alerts': self.alert_manager.get_alerts(limit=10)
        }
    
    def start_monitoring(self, check_interval: float = 60.0):
        """Start background monitoring loop."""
        self.running = True
        
        def monitor_loop():
            while self.running:
                # Score all users periodically
                user_features = self.feature_buffer.get_all_user_features()
                scores = self.scorer.score_batch(user_features)
                alerts = self.alert_manager.process_scores(scores)
                
                if alerts:
                    logger.info(f"Generated {len(alerts)} new alerts")
                
                time.sleep(check_interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info(f"Started real-time monitoring (interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring loop."""
        self.running = False


# Simple REST API using Flask (if available)
def create_api_server(detector: RealTimeDetector):
    """
    Create Flask API server for real-time detection.
    
    Endpoints:
        POST /event - Submit a new event
        GET /alerts - Get current alerts
        GET /dashboard - Get dashboard data
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.warning("Flask not installed. API server not available.")
        logger.warning("Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    
    @app.route('/event', methods=['POST'])
    def process_event():
        event = request.json
        alert = detector.process_event(event)
        return jsonify({'alert': alert})
    
    @app.route('/alerts', methods=['GET'])
    def get_alerts():
        status = request.args.get('status')
        alerts = detector.alert_manager.get_alerts(status=status)
        return jsonify({'alerts': alerts})
    
    @app.route('/dashboard', methods=['GET'])
    def get_dashboard():
        data = detector.get_dashboard_data()
        return jsonify(data)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy'})
    
    return app


def demo_realtime_detection():
    """Demonstrate real-time detection with simulated events."""
    logger.info("=" * 80)
    logger.info("REAL-TIME DETECTION DEMO")
    logger.info("=" * 80)
    
    detector = RealTimeDetector()
    
    # Simulate events
    logger.info("\nSimulating event stream...")
    
    sample_users = ['USER001', 'USER002', 'USER003', 'INSIDER001']
    
    for i in range(50):
        user = np.random.choice(sample_users)
        
        # Insider has more after-hours activity
        if user == 'INSIDER001':
            hour = np.random.choice([1, 2, 3, 22, 23])
        else:
            hour = np.random.randint(8, 18)
        
        event = {
            'user_id': user,
            'timestamp': datetime.now().isoformat(),
            'activity_type': np.random.choice(['logon', 'file', 'email']),
            'hour': hour
        }
        
        alert = detector.process_event(event)
        if alert:
            logger.info(f"  ALERT: {alert['user_id']} - {alert['severity']} ({alert['score']:.2f})")
    
    # Summary
    dashboard = detector.get_dashboard_data()
    logger.info(f"\n✓ Demo complete")
    logger.info(f"  Monitored users: {dashboard['monitored_users']}")
    logger.info(f"  Total alerts: {dashboard['total_alerts']}")
    
    # Save sample output
    output_path = detector.output_dir / 'demo_output.json'
    with open(output_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    logger.info(f"  Output saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='Start REST API server')
    parser.add_argument('--port', type=int, default=8080, help='API server port')
    parser.add_argument('--demo', action='store_true', help='Run demo simulation')
    args = parser.parse_args()
    
    if args.serve:
        detector = RealTimeDetector()
        detector.start_monitoring()
        app = create_api_server(detector)
        if app:
            logger.info(f"Starting API server on port {args.port}")
            app.run(host='0.0.0.0', port=args.port)
    elif args.demo:
        demo_realtime_detection()
    else:
        demo_realtime_detection()
