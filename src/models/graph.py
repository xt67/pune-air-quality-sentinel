"""
Graph construction utilities for ST-GNN.

Builds spatial graphs from station coordinates using distance-based adjacency.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

# Pune station coordinates (latitude, longitude)
PUNE_STATIONS: Dict[str, Tuple[float, float]] = {
    "karve_road": (18.5018, 73.8170),       # MH020
    "shivajinagar": (18.5314, 73.8446),     # MH021
    "hadapsar": (18.5089, 73.9260),         # MH022
    "katraj": (18.4575, 73.8678),
    "nigdi": (18.6520, 73.7680),
    "bhosari": (18.6298, 73.8483),
    "pimpri": (18.6186, 73.8037),
    "kothrud": (18.5074, 73.8077),
    "viman_nagar": (18.5679, 73.9143),
    "aundh": (18.5590, 73.8076),
}


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        coord1: (latitude, longitude) of first point
        coord2: (latitude, longitude) of second point
        
    Returns:
        Distance in kilometers
    """
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in km
    r = 6371.0
    return r * c


def build_adjacency_matrix(
    stations: Dict[str, Tuple[float, float]],
    threshold_km: float = 10.0,
    sigma: float = 5.0,
    self_loop: bool = True,
) -> np.ndarray:
    """
    Build a weighted adjacency matrix using Gaussian kernel on distances.
    
    Edges are created between stations within threshold_km of each other.
    Edge weights are computed as: exp(-d^2 / (2 * sigma^2))
    
    Args:
        stations: Dict mapping station name to (lat, lon)
        threshold_km: Maximum distance for edge creation
        sigma: Gaussian kernel bandwidth parameter
        self_loop: Whether to add self-loops (diagonal = 1)
        
    Returns:
        Adjacency matrix of shape (num_stations, num_stations)
    """
    station_names = list(stations.keys())
    n = len(station_names)
    adj = np.zeros((n, n), dtype=np.float32)
    
    for i, name_i in enumerate(station_names):
        for j, name_j in enumerate(station_names):
            if i == j:
                adj[i, j] = 1.0 if self_loop else 0.0
            else:
                dist = haversine_distance(stations[name_i], stations[name_j])
                if dist <= threshold_km:
                    # Gaussian kernel weight
                    adj[i, j] = math.exp(-(dist**2) / (2 * sigma**2))
    
    return adj


def build_edge_index(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert adjacency matrix to edge index format for PyTorch Geometric.
    
    Args:
        adj: Adjacency matrix of shape (N, N)
        
    Returns:
        Tuple of (edge_index, edge_weight) where:
        - edge_index: (2, num_edges) array of [source, target] indices
        - edge_weight: (num_edges,) array of edge weights
    """
    sources, targets = np.where(adj > 0)
    edge_index = np.stack([sources, targets], axis=0)
    edge_weight = adj[sources, targets]
    return edge_index, edge_weight


def get_distance_matrix(stations: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Compute pairwise distance matrix between all stations.
    
    Args:
        stations: Dict mapping station name to (lat, lon)
        
    Returns:
        Distance matrix of shape (num_stations, num_stations) in km
    """
    station_names = list(stations.keys())
    n = len(station_names)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i, name_i in enumerate(station_names):
        for j, name_j in enumerate(station_names):
            if i != j:
                dist_matrix[i, j] = haversine_distance(stations[name_i], stations[name_j])
    
    return dist_matrix


def normalize_adjacency(adj: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """
    Normalize adjacency matrix for GCN.
    
    Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    Row normalization: D^(-1) * A
    
    Args:
        adj: Adjacency matrix
        symmetric: If True, use symmetric normalization
        
    Returns:
        Normalized adjacency matrix
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    
    # Degree matrix
    d = np.sum(adj, axis=1)
    
    if symmetric:
        d_inv_sqrt = np.power(d + eps, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        return d_mat @ adj @ d_mat
    else:
        d_inv = np.power(d + eps, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = np.diag(d_inv)
        return d_mat @ adj


def save_graph_data(
    output_path: str,
    adj: np.ndarray,
    station_names: List[str],
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save graph data to npz file.
    
    Args:
        output_path: Path to save the npz file
        adj: Adjacency matrix
        station_names: List of station names (in order)
        metadata: Optional metadata dict
    """
    edge_index, edge_weight = build_edge_index(adj)
    
    save_dict = {
        "adjacency": adj,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "station_names": np.array(station_names, dtype=object),
    }
    
    if metadata:
        for key, value in metadata.items():
            save_dict[f"meta_{key}"] = np.array(value)
    
    np.savez_compressed(output_path, **save_dict)


def load_graph_data(path: str) -> Dict[str, np.ndarray]:
    """
    Load graph data from npz file.
    
    Args:
        path: Path to the npz file
        
    Returns:
        Dict with adjacency, edge_index, edge_weight, station_names
    """
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
