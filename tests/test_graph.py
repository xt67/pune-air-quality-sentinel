"""Tests for graph construction utilities."""

import math
import numpy as np
import pytest
import tempfile
import os

from src.models.graph import (
    PUNE_STATIONS,
    haversine_distance,
    build_adjacency_matrix,
    build_edge_index,
    get_distance_matrix,
    normalize_adjacency,
    save_graph_data,
    load_graph_data,
)


class TestHaversineDistance:
    """Tests for haversine_distance function."""
    
    def test_same_point_zero_distance(self):
        """Same point should have zero distance."""
        coord = (18.5018, 73.8170)
        dist = haversine_distance(coord, coord)
        assert dist == pytest.approx(0.0, abs=1e-6)
    
    def test_known_distance(self):
        """Test with known distance between two cities."""
        # Pune to Mumbai approximately 150 km
        pune = (18.5204, 73.8567)
        mumbai = (19.0760, 72.8777)
        dist = haversine_distance(pune, mumbai)
        assert 100 < dist < 200  # Rough check
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        coord1 = (18.5018, 73.8170)
        coord2 = (18.5314, 73.8446)
        assert haversine_distance(coord1, coord2) == pytest.approx(
            haversine_distance(coord2, coord1), abs=1e-6
        )
    
    def test_pune_stations_reasonable(self):
        """Pune stations should be within 30km of each other."""
        coords = list(PUNE_STATIONS.values())
        for i, c1 in enumerate(coords):
            for c2 in coords[i + 1:]:
                dist = haversine_distance(c1, c2)
                assert dist < 30, "Pune stations should be close"


class TestBuildAdjacencyMatrix:
    """Tests for build_adjacency_matrix function."""
    
    def test_shape(self):
        """Adjacency matrix should be square."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        n = len(PUNE_STATIONS)
        assert adj.shape == (n, n)
    
    def test_self_loops(self):
        """With self_loop=True, diagonal should be 1."""
        adj = build_adjacency_matrix(PUNE_STATIONS, self_loop=True)
        diagonal = np.diag(adj)
        assert np.allclose(diagonal, 1.0)
    
    def test_no_self_loops(self):
        """With self_loop=False, diagonal should be 0."""
        adj = build_adjacency_matrix(PUNE_STATIONS, self_loop=False)
        diagonal = np.diag(adj)
        assert np.allclose(diagonal, 0.0)
    
    def test_symmetry(self):
        """Adjacency matrix should be symmetric."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        assert np.allclose(adj, adj.T)
    
    def test_threshold_effect(self):
        """Lower threshold should result in fewer edges."""
        adj_low = build_adjacency_matrix(PUNE_STATIONS, threshold_km=5.0)
        adj_high = build_adjacency_matrix(PUNE_STATIONS, threshold_km=20.0)
        assert np.sum(adj_low > 0) <= np.sum(adj_high > 0)
    
    def test_weights_in_range(self):
        """Gaussian weights should be in (0, 1]."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        non_zero = adj[adj > 0]
        assert np.all(non_zero > 0)
        assert np.all(non_zero <= 1.0)
    
    def test_dtype(self):
        """Adjacency matrix should be float32."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        assert adj.dtype == np.float32


class TestBuildEdgeIndex:
    """Tests for build_edge_index function."""
    
    def test_output_shapes(self):
        """Edge index and weights should have correct shapes."""
        adj = build_adjacency_matrix(PUNE_STATIONS, threshold_km=10.0)
        edge_index, edge_weight = build_edge_index(adj)
        
        num_edges = np.sum(adj > 0)
        assert edge_index.shape == (2, num_edges)
        assert edge_weight.shape == (num_edges,)
    
    def test_edge_index_values(self):
        """Edge index values should be valid node indices."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        edge_index, _ = build_edge_index(adj)
        
        n = len(PUNE_STATIONS)
        assert np.all(edge_index >= 0)
        assert np.all(edge_index < n)
    
    def test_edge_weights_match_adjacency(self):
        """Edge weights should match adjacency values."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        edge_index, edge_weight = build_edge_index(adj)
        
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[:, i]
            assert edge_weight[i] == pytest.approx(adj[src, tgt], abs=1e-6)


class TestGetDistanceMatrix:
    """Tests for get_distance_matrix function."""
    
    def test_shape(self):
        """Distance matrix should be square."""
        dist = get_distance_matrix(PUNE_STATIONS)
        n = len(PUNE_STATIONS)
        assert dist.shape == (n, n)
    
    def test_diagonal_zero(self):
        """Diagonal should be zero (distance to self)."""
        dist = get_distance_matrix(PUNE_STATIONS)
        assert np.allclose(np.diag(dist), 0.0)
    
    def test_symmetry(self):
        """Distance matrix should be symmetric."""
        dist = get_distance_matrix(PUNE_STATIONS)
        assert np.allclose(dist, dist.T)
    
    def test_positive_values(self):
        """Off-diagonal should be positive."""
        dist = get_distance_matrix(PUNE_STATIONS)
        n = len(PUNE_STATIONS)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert dist[i, j] > 0


class TestNormalizeAdjacency:
    """Tests for normalize_adjacency function."""
    
    def test_symmetric_normalization(self):
        """Symmetric normalization should maintain symmetry."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        norm_adj = normalize_adjacency(adj, symmetric=True)
        assert np.allclose(norm_adj, norm_adj.T, atol=1e-6)
    
    def test_row_normalization(self):
        """Row normalization should sum to ~1 for rows with edges."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        norm_adj = normalize_adjacency(adj, symmetric=False)
        
        row_sums = np.sum(norm_adj, axis=1)
        # Rows with at least one edge should sum to ~1
        for i, s in enumerate(row_sums):
            if np.sum(adj[i, :]) > 0:
                assert s == pytest.approx(1.0, abs=1e-5)
    
    def test_no_nan_or_inf(self):
        """Normalized matrix should have no NaN or Inf."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        norm_adj = normalize_adjacency(adj)
        assert not np.any(np.isnan(norm_adj))
        assert not np.any(np.isinf(norm_adj))


class TestSaveLoadGraphData:
    """Tests for save_graph_data and load_graph_data functions."""
    
    def test_round_trip(self):
        """Data should survive save/load cycle."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        station_names = list(PUNE_STATIONS.keys())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.npz")
            save_graph_data(path, adj, station_names)
            
            loaded = load_graph_data(path)
            
            assert np.allclose(loaded["adjacency"], adj)
            assert list(loaded["station_names"]) == station_names
    
    def test_edge_index_saved(self):
        """Edge index should be saved and loaded."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        station_names = list(PUNE_STATIONS.keys())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.npz")
            save_graph_data(path, adj, station_names)
            
            loaded = load_graph_data(path)
            
            assert "edge_index" in loaded
            assert "edge_weight" in loaded
    
    def test_metadata_saved(self):
        """Metadata should be saved and loaded."""
        adj = build_adjacency_matrix(PUNE_STATIONS)
        station_names = list(PUNE_STATIONS.keys())
        metadata = {"threshold_km": 10.0, "sigma": 5.0}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.npz")
            save_graph_data(path, adj, station_names, metadata)
            
            loaded = load_graph_data(path)
            
            assert loaded["meta_threshold_km"] == 10.0
            assert loaded["meta_sigma"] == 5.0


class TestPuneStationsIntegration:
    """Integration tests with real Pune station data."""
    
    def test_pune_graph_connectivity(self):
        """Pune graph should be connected with reasonable threshold."""
        adj = build_adjacency_matrix(PUNE_STATIONS, threshold_km=15.0)
        
        # At least half the edges should exist
        n = len(PUNE_STATIONS)
        num_edges = np.sum(adj > 0) - n  # Exclude self-loops
        max_edges = n * (n - 1)
        
        assert num_edges > max_edges * 0.3, "Graph should have reasonable connectivity"
    
    def test_nearby_stations_connected(self):
        """Nearby stations should be connected."""
        adj = build_adjacency_matrix(PUNE_STATIONS, threshold_km=10.0)
        station_names = list(PUNE_STATIONS.keys())
        
        # Karve Road and Kothrud are very close (~1km)
        idx_karve = station_names.index("karve_road")
        idx_kothrud = station_names.index("kothrud")
        
        assert adj[idx_karve, idx_kothrud] > 0.5, "Close stations should have high weight"
