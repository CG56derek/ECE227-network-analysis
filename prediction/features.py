from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from data_utils import load_all_graphs, resolve_data_dir


def check_contiguous_node_labels(G: nx.Graph) -> None:
    nodes = sorted(G.nodes())
    expected = list(range(G.number_of_nodes()))
    if nodes != expected:
        raise ValueError(
            "Node labels are not contiguous 0..N-1. "
            "Please relabel the graph before building features."
        )


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)


    std_safe = np.where(std < 1e-12, 1.0, std)
    X_std = (X - mean) / std_safe

    return X_std.astype(np.float32), mean.astype(np.float32), std_safe.astype(np.float32)


def compute_log_degree(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    values = np.zeros(n, dtype=np.float32)

    for node, deg in G.degree():
        values[node] = np.log1p(deg)

    return values


def compute_clustering_coefficient(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    values = np.zeros(n, dtype=np.float32)

    clustering = nx.clustering(G)
    for node, val in clustering.items():
        values[node] = float(val)

    return values


def compute_average_neighbor_degree(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    values = np.zeros(n, dtype=np.float32)

    avg_nd = nx.average_neighbor_degree(G)
    for node, val in avg_nd.items():
        values[node] = float(val)

    return values


def compute_core_number(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    values = np.zeros(n, dtype=np.float32)

    core = nx.core_number(G)
    for node, val in core.items():
        values[node] = float(val)

    return values


def build_feature_matrix(
    G: nx.Graph,
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    check_contiguous_node_labels(G)

    feature_names = [
        "log_degree",
        "clustering",
        "avg_neighbor_degree",
        "core_number",
    ]

    raw_features = {
        "log_degree": compute_log_degree(G),
        "clustering": compute_clustering_coefficient(G),
        "avg_neighbor_degree": compute_average_neighbor_degree(G),
        "core_number": compute_core_number(G),
    }

    X = np.column_stack([raw_features[name] for name in feature_names]).astype(np.float32)

    if standardize:
        X, _, _ = standardize_features(X)

    return X, feature_names, raw_features


def summarize_feature_matrix(
    X: np.ndarray,
    feature_names: List[str],
    name: str = "Graph",
) -> None:
    print(f"\n[{name} feature summary]")
    print(f"shape = {X.shape}")

    for i, feat_name in enumerate(feature_names):
        col = X[:, i]
        print(
            f"{feat_name:<20} "
            f"mean={col.mean():>8.4f} "
            f"std={col.std():>8.4f} "
            f"min={col.min():>8.4f} "
            f"max={col.max():>8.4f}"
        )


def build_all_feature_matrices(
    data_dir: Path | str | None = None,
    use_gcc: bool = True,
    relabel: bool = True,
    standardize: bool = True,
) -> Dict[str, Dict[str, object]]:
    if data_dir is None:
        data_dir = resolve_data_dir()

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=use_gcc, relabel=relabel)

    results: Dict[str, Dict[str, object]] = {}

    for name, G in graphs.items():
        X, feature_names, raw_features = build_feature_matrix(G, standardize=standardize)

        results[name] = {
            "graph": G,
            "X": X,
            "feature_names": feature_names,
            "raw_features": raw_features,
        }

    return results


def save_feature_matrix(
    save_path: Path | str,
    X: np.ndarray,
    feature_names: List[str],
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        X=X.astype(np.float32),
        feature_names=np.array(feature_names, dtype=object),
    )


def save_all_feature_matrices(
    results: Dict[str, Dict[str, object]],
    out_dir: Path | str = "artifacts/features",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, item in results.items():
        X = item["X"]
        feature_names = item["feature_names"]
        save_path = out_dir / f"{name.lower()}_features.npz"
        save_feature_matrix(save_path, X, feature_names)
        print(f"Saved {name} features to: {save_path}")


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    results = build_all_feature_matrices(
        data_dir=data_dir,
        use_gcc=True,
        relabel=True,
        standardize=True,
    )

    for name, item in results.items():
        G = item["graph"]
        X = item["X"]
        feature_names = item["feature_names"]

        print(f"\n{name}")
        print(f"nodes = {G.number_of_nodes()}, edges = {G.number_of_edges()}")
        summarize_feature_matrix(X, feature_names, name=name)

    save_all_feature_matrices(results, out_dir="artifacts/features")


if __name__ == "__main__":
    main()
