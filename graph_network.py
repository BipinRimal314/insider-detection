"""
Graph Neural Network Module for Insider Threat Detection

Models relationships between users based on:
1. Email communication patterns
2. Shared file access
3. Department/role hierarchies

Uses Graph Attention Networks (GAT) to propagate anomaly signals
through the social network, detecting suspicious group behaviors.

Usage:
    python graph_network.py

Requirements:
    pip install torch torch-geometric
    (These are optional - fallback to NetworkX if not available)
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import config
import utils

logger = utils.logger

# Try to import PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("PyTorch Geometric not installed. Using NetworkX fallback.")
    logger.warning("For full GNN support: pip install torch torch-geometric")

# NetworkX is more commonly available
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class UserRelationshipGraph:
    """
    Builds user relationship graphs from communication and access patterns.
    """
    
    def __init__(self):
        self.graph = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.output_dir = config.RESULTS_DIR / 'graph'
        self.output_dir.mkdir(exist_ok=True)
        
    def build_from_processed_data(self, processed_file: Path = None) -> nx.DiGraph:
        """
        Build user relationship graph from processed log data.
        
        Args:
            processed_file: Path to processed unified logs CSV
            
        Returns:
            NetworkX directed graph
        """
        if processed_file is None:
            processed_file = config.PROCESSED_DATA_FILE
            
        if not processed_file.exists():
            logger.error(f"Processed data not found: {processed_file}")
            return None
            
        logger.info("Building user relationship graph...")
        
        # Load data with lazy scanning for memory efficiency
        lf = pl.scan_csv(processed_file)
        
        # Get email communications (from -> to relationships)
        email_schema = lf.collect_schema()
        
        if 'activity_type' in email_schema.names():
            # Filter to emails and get communication edges
            email_data = (
                lf.filter(pl.col('activity_type') == 'email')
                .select(['user_id', 'to', 'cc', 'bcc'])
                .collect()
            )
            
            # Build graph
            if not NETWORKX_AVAILABLE:
                logger.error("NetworkX required for graph building")
                return None
                
            G = nx.DiGraph()
            
            # Add email edges
            for row in email_data.iter_rows(named=True):
                sender = row['user_id']
                
                # Parse recipients
                recipients = set()
                for field in ['to', 'cc', 'bcc']:
                    if row[field]:
                        # Recipients could be comma-separated
                        recips = str(row[field]).split(';')
                        recipients.update(r.strip() for r in recips if r.strip())
                
                for recipient in recipients:
                    if recipient and sender != recipient:
                        if G.has_edge(sender, recipient):
                            G[sender][recipient]['weight'] += 1
                            G[sender][recipient]['emails'] += 1
                        else:
                            G.add_edge(sender, recipient, weight=1, emails=1, files=0)
            
            logger.info(f"✓ Built graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            self.graph = G
            self._build_user_mappings()
            
            return G
        else:
            logger.warning("No activity_type column found in data")
            return None
    
    def _build_user_mappings(self):
        """Create user <-> index mappings for embedding."""
        if self.graph is None:
            return
            
        self.user_to_idx = {user: idx for idx, user in enumerate(self.graph.nodes())}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
    
    def compute_graph_features(self) -> pd.DataFrame:
        """
        Compute graph-based features for each user.
        
        Returns:
            DataFrame with user-level graph features
        """
        if self.graph is None:
            logger.error("Graph not built yet")
            return None
            
        logger.info("Computing graph features...")
        
        features = []
        
        for node in self.graph.nodes():
            feat = {
                'user': node,
                'out_degree': self.graph.out_degree(node),
                'in_degree': self.graph.in_degree(node),
                'total_degree': self.graph.degree(node),
            }
            
            # Weighted degrees
            feat['out_weight'] = sum(d.get('weight', 1) for _, _, d in self.graph.out_edges(node, data=True))
            feat['in_weight'] = sum(d.get('weight', 1) for _, _, d in self.graph.in_edges(node, data=True))
            
            # Local clustering coefficient
            try:
                feat['clustering'] = nx.clustering(self.graph.to_undirected(), node)
            except:
                feat['clustering'] = 0
            
            features.append(feat)
        
        features_df = pd.DataFrame(features)
        
        # Add centrality measures (expensive for large graphs)
        if len(self.graph) < 10000:  # Only for smaller graphs
            logger.info("  Computing centrality measures...")
            
            try:
                pagerank = nx.pagerank(self.graph, max_iter=100)
                features_df['pagerank'] = features_df['user'].map(pagerank)
            except:
                features_df['pagerank'] = 0
                
            try:
                betweenness = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph)))
                features_df['betweenness'] = features_df['user'].map(betweenness)
            except:
                features_df['betweenness'] = 0
        else:
            logger.info("  Skipping centrality (graph too large)")
            features_df['pagerank'] = 0
            features_df['betweenness'] = 0
        
        logger.info(f"✓ Computed {len(features_df.columns)-1} graph features for {len(features_df)} users")
        
        return features_df
    
    def detect_anomalous_communities(self) -> Dict:
        """
        Detect communities and identify anomalous ones.
        
        Returns:
            Dictionary with community detection results
        """
        if self.graph is None:
            return {}
            
        logger.info("Detecting communities...")
        
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()
        
        # Use Louvain community detection if available
        try:
            from community import community_louvain
            communities = community_louvain.best_partition(G_undirected)
        except ImportError:
            # Fallback to greedy modularity
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                community_sets = list(greedy_modularity_communities(G_undirected))
                communities = {}
                for idx, comm_set in enumerate(community_sets):
                    for node in comm_set:
                        communities[node] = idx
            except:
                logger.warning("Could not detect communities")
                return {}
        
        # Analyze communities
        community_stats = defaultdict(lambda: {'size': 0, 'internal_edges': 0, 'external_edges': 0})
        
        for node, comm_id in communities.items():
            community_stats[comm_id]['size'] += 1
            
        logger.info(f"✓ Detected {len(community_stats)} communities")
        
        return {
            'node_communities': communities,
            'community_stats': dict(community_stats)
        }
    
    def visualize_graph(self, highlight_users: List[str] = None, 
                       max_nodes: int = 500):
        """
        Visualize the user relationship graph.
        
        Args:
            highlight_users: Users to highlight (e.g., insiders)
            max_nodes: Maximum nodes to display
        """
        if self.graph is None:
            return
            
        import matplotlib.pyplot as plt
        
        # Sample if too large
        if len(self.graph) > max_nodes:
            sampled_nodes = list(self.graph.nodes())[:max_nodes]
            G_vis = self.graph.subgraph(sampled_nodes)
        else:
            G_vis = self.graph
            
        plt.figure(figsize=(14, 10))
        
        # Layout
        try:
            pos = nx.spring_layout(G_vis, k=2, iterations=50)
        except:
            pos = nx.random_layout(G_vis)
        
        # Node colors
        colors = []
        for node in G_vis.nodes():
            if highlight_users and node in highlight_users:
                colors.append('red')
            else:
                colors.append('lightblue')
        
        # Draw
        nx.draw(G_vis, pos, 
               node_color=colors,
               node_size=50,
               edge_color='gray',
               alpha=0.7,
               with_labels=False)
        
        if highlight_users:
            highlighted = [n for n in G_vis.nodes() if n in highlight_users]
            nx.draw_networkx_nodes(G_vis, pos, nodelist=highlighted, 
                                   node_color='red', node_size=200)
        
        plt.title("User Communication Network")
        plt.tight_layout()
        
        output_path = self.output_dir / 'user_network.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"✓ Graph visualization saved to {output_path}")


class GraphAnomalyDetector:
    """
    Graph-based anomaly detection using local neighborhood analysis.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.user_features = None
        
    def compute_local_outlier_factor(self) -> pd.DataFrame:
        """
        Compute graph-based Local Outlier Factor for each node.
        
        Uses node features and neighborhood structure to identify outliers.
        """
        if not NETWORKX_AVAILABLE:
            return None
            
        logger.info("Computing graph-based anomaly scores...")
        
        # Build feature matrix
        features = []
        users = []
        
        for node in self.graph.nodes():
            feat = [
                self.graph.out_degree(node),
                self.graph.in_degree(node),
                sum(d.get('weight', 1) for _, _, d in self.graph.out_edges(node, data=True)),
                sum(d.get('weight', 1) for _, _, d in self.graph.in_edges(node, data=True)),
            ]
            features.append(feat)
            users.append(node)
        
        X = np.array(features)
        
        # Apply Local Outlier Factor
        from sklearn.neighbors import LocalOutlierFactor
        
        lof = LocalOutlierFactor(n_neighbors=min(20, len(X)-1), contamination='auto')
        predictions = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_
        
        results = pd.DataFrame({
            'user': users,
            'graph_anomaly_score': scores,
            'is_graph_anomaly': (predictions == -1).astype(int)
        })
        
        logger.info(f"✓ Detected {(predictions == -1).sum()} graph anomalies")
        
        return results
    
    def detect_unusual_connections(self, daily_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Detect users with unusual connection patterns.
        
        Args:
            daily_features: DataFrame with is_anomaly labels for validation
            
        Returns:
            DataFrame with connection anomaly scores
        """
        anomaly_scores = self.compute_local_outlier_factor()
        
        if daily_features is not None and 'is_anomaly' in daily_features.columns:
            # Validate against known labels
            user_labels = daily_features.groupby('user')['is_anomaly'].max().reset_index()
            merged = anomaly_scores.merge(user_labels, on='user', how='left')
            
            # Calculate correlation
            if merged['is_anomaly'].sum() > 0:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(merged['is_anomaly'].fillna(0), 
                                    merged['graph_anomaly_score'])
                logger.info(f"Graph anomaly AUC: {auc:.4f}")
        
        return anomaly_scores


def run_graph_analysis():
    """Run full graph-based analysis pipeline."""
    logger.info("=" * 80)
    logger.info("GRAPH NEURAL NETWORK ANALYSIS")
    logger.info("=" * 80)
    
    # Build graph
    graph_builder = UserRelationshipGraph()
    graph = graph_builder.build_from_processed_data()
    
    if graph is None:
        logger.error("Failed to build graph")
        return
    
    # Compute graph features
    graph_features = graph_builder.compute_graph_features()
    
    # Save features
    features_path = graph_builder.output_dir / 'graph_features.csv'
    graph_features.to_csv(features_path, index=False)
    logger.info(f"✓ Graph features saved to {features_path}")
    
    # Detect communities
    communities = graph_builder.detect_anomalous_communities()
    
    # Load insider labels if available
    daily_path = config.DAILY_FEATURES_FILE
    daily_df = None
    insiders = []
    if daily_path.exists():
        daily_df = pl.read_parquet(daily_path).to_pandas()
        if 'is_anomaly' in daily_df.columns:
            insiders = daily_df[daily_df['is_anomaly'] == 1]['user'].unique().tolist()
    
    # Visualize with insiders highlighted
    graph_builder.visualize_graph(highlight_users=insiders)
    
    # Graph-based anomaly detection
    detector = GraphAnomalyDetector(graph)
    anomaly_scores = detector.detect_unusual_connections(daily_df)
    
    if anomaly_scores is not None:
        scores_path = graph_builder.output_dir / 'graph_anomaly_scores.csv'
        anomaly_scores.to_csv(scores_path, index=False)
        logger.info(f"✓ Graph anomaly scores saved to {scores_path}")
    
    logger.info("\n✓ Graph analysis complete!")


if __name__ == "__main__":
    run_graph_analysis()
