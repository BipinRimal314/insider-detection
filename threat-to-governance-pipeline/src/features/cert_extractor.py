"""Extract UBFS features from CMU-CERT dataset.

Maps the daily behavioural features from the MSc thesis pipeline
into the Unified Behavioural Feature Schema. This extractor
operates on the output of the existing feature engineering
code (daily_features DataFrame).

Feature Mapping:
    CMU-CERT daily features (~21 columns per user-day)
    → UBFS vector (20 dimensions)

Some UBFS slots require derived features not directly present
in the CMU-CERT output (e.g., peer_distance, sequence_entropy).
These are computed here from the daily features DataFrame.
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from .ubfs_schema import (
    FEATURE_DEFINITIONS,
    FeatureCategory,
    UBFSConfig,
    UBFSVector,
    ubfs_feature_names,
)


class CERTFeatureExtractor:
    """Extracts UBFS vectors from CMU-CERT daily features.

    Expects input from the MSc feature engineering pipeline:
    a Polars DataFrame with columns like logon_count,
    after_hours_logons, unique_pcs, etc.
    """

    def __init__(self):
        self.config = UBFSConfig()
        self._peer_baselines: Optional[Dict[str, np.ndarray]] = None

    def extract_user_day(
        self, row: Dict, peer_baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract a single UBFS vector from one user-day row.

        Args:
            row: Dictionary of daily feature values for one user-day.
            peer_baseline: Mean UBFS vector for the user's peer group
                (used for deviation features). None = use zeros.

        Returns:
            UBFS vector as numpy array of shape (total_dim,).
        """
        features = np.zeros(self.config.total_dim, dtype=np.float32)
        slices = self.config.category_slices

        # TEMPORAL
        s = slices[FeatureCategory.TEMPORAL]
        features[s.start + 0] = row.get("first_logon_hour", 0.0)
        last_logoff = row.get("last_logoff_hour", 0.0)
        first_logon = row.get("first_logon_hour", 0.0)
        features[s.start + 1] = max(last_logoff - first_logon, 0.0)
        logon_count = row.get("logon_count", 1)
        ah_logons = row.get("after_hours_logons", 0)
        features[s.start + 2] = (
            ah_logons / max(logon_count, 1)
        )
        features[s.start + 3] = float(
            row.get("day_of_week", 0) >= 5
        )

        # FREQUENCY
        s = slices[FeatureCategory.FREQUENCY]
        features[s.start + 0] = row.get("logon_count", 0.0)
        features[s.start + 1] = row.get("emails_sent", 0.0)
        features[s.start + 2] = row.get("device_activity", 0.0)
        # Event rate z-score filled during batch normalisation
        features[s.start + 3] = 0.0

        # VOLUME
        s = slices[FeatureCategory.VOLUME]
        features[s.start + 0] = row.get("attachment_size", 0.0)
        features[s.start + 1] = row.get("total_recipients", 0.0)
        features[s.start + 2] = 0.0  # Std computed at batch level

        # SCOPE
        s = slices[FeatureCategory.SCOPE]
        features[s.start + 0] = row.get("unique_pcs", 0.0)
        features[s.start + 1] = row.get("unique_domains", 0.0)
        pc_ratio = (
            row.get("unique_pcs", 0) / max(logon_count, 1)
        )
        features[s.start + 2] = pc_ratio

        # SEQUENCE (requires temporal context — placeholder)
        s = slices[FeatureCategory.SEQUENCE]
        features[s.start + 0] = 0.0  # Filled by extract_batch
        features[s.start + 1] = 0.0
        features[s.start + 2] = 0.0

        # DEVIATION
        s = slices[FeatureCategory.DEVIATION]
        if peer_baseline is not None:
            features[s.start + 0] = float(
                np.linalg.norm(features[:s.start] - peer_baseline[:s.start])
            )
        features[s.start + 1] = 0.0  # Filled by extract_batch

        # PRIVILEGE
        s = slices[FeatureCategory.PRIVILEGE]
        features[s.start + 0] = 0.0  # CMU-CERT lacks explicit roles

        return features

    def extract_batch(
        self, df: pl.DataFrame
    ) -> tuple[np.ndarray, List[str], List[str]]:
        """Extract UBFS vectors for all user-days in a DataFrame.

        Also computes batch-level features: event rate z-scores,
        volume variability, sequence entropy (from action type
        distributions), and peer group distances.

        Args:
            df: Daily features DataFrame with columns user_id,
                date, and feature columns.

        Returns:
            Tuple of:
                X: UBFS matrix (n_samples, ubfs_dim)
                entity_ids: List of user_ids
                timestamps: List of date strings
        """
        rows = df.to_dicts()
        n = len(rows)
        X = np.zeros((n, self.config.total_dim), dtype=np.float32)
        entity_ids = []
        timestamps = []

        # First pass: extract per-row features
        for i, row in enumerate(rows):
            X[i] = self.extract_user_day(row)
            entity_ids.append(str(row.get("user_id", f"user_{i}")))
            timestamps.append(str(row.get("date", "")))

        # Compute batch-level derived features
        X = self._compute_batch_features(X, df)

        return X, entity_ids, timestamps

    def _compute_batch_features(
        self, X: np.ndarray, df: pl.DataFrame
    ) -> np.ndarray:
        """Fill in features that require batch-level statistics."""
        slices = self.config.category_slices

        # Event rate z-score (FREQUENCY slot 3)
        freq_s = slices[FeatureCategory.FREQUENCY]
        total_events = X[:, freq_s.start] + X[:, freq_s.start + 1]
        mean_rate = np.mean(total_events)
        std_rate = np.std(total_events)
        if std_rate > 0:
            X[:, freq_s.start + 3] = (
                (total_events - mean_rate) / std_rate
            )

        # Volume variability (VOLUME slot 2)
        vol_s = slices[FeatureCategory.VOLUME]
        # Per-user std of attachment sizes
        if "attachment_size" in df.columns:
            user_stds = (
                df.group_by("user_id")
                .agg(pl.col("attachment_size").std().alias("vol_std"))
            )
            std_map = dict(
                zip(
                    user_stds["user_id"].to_list(),
                    user_stds["vol_std"].to_list(),
                )
            )
            rows = df.to_dicts()
            for i, row in enumerate(rows):
                uid = row.get("user_id", "")
                X[i, vol_s.start + 2] = std_map.get(uid, 0.0) or 0.0

        # Sequence entropy (SEQUENCE slot 0)
        # Approximate from action type distribution per user
        seq_s = slices[FeatureCategory.SEQUENCE]
        for i in range(len(X)):
            action_counts = [
                X[i, freq_s.start + j]
                for j in range(3)
                if X[i, freq_s.start + j] > 0
            ]
            if action_counts:
                total = sum(action_counts)
                probs = [c / total for c in action_counts]
                entropy = -sum(
                    p * np.log2(p) for p in probs if p > 0
                )
                X[i, seq_s.start + 0] = entropy

        # Peer group distance (DEVIATION slot 0)
        dev_s = slices[FeatureCategory.DEVIATION]
        global_mean = np.mean(X[:, :dev_s.start], axis=0)
        for i in range(len(X)):
            X[i, dev_s.start + 0] = float(
                np.linalg.norm(X[i, :dev_s.start] - global_mean)
            )

        # Self-deviation (DEVIATION slot 1)
        # Approximate: distance from user's own mean
        rows = df.to_dicts()
        user_indices: Dict[str, List[int]] = {}
        for i, row in enumerate(rows):
            uid = str(row.get("user_id", ""))
            user_indices.setdefault(uid, []).append(i)

        for uid, indices in user_indices.items():
            if len(indices) < 2:
                continue
            user_vecs = X[indices, :dev_s.start]
            user_mean = np.mean(user_vecs, axis=0)
            for idx in indices:
                X[idx, dev_s.start + 1] = float(
                    np.linalg.norm(
                        X[idx, :dev_s.start] - user_mean
                    )
                )

        return X

    def to_ubfs_vectors(
        self,
        X: np.ndarray,
        entity_ids: List[str],
        timestamps: List[str],
    ) -> List[UBFSVector]:
        """Wrap raw arrays into UBFSVector objects."""
        return [
            UBFSVector(
                values=X[i],
                entity_id=entity_ids[i],
                domain="cert",
                timestamp=timestamps[i],
            )
            for i in range(len(X))
        ]
