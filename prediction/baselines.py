from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data_utils import load_all_graphs, resolve_data_dir


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def reconstruct_train_graph(train_graph_edges: np.ndarray, num_nodes: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(train_graph_edges.tolist())
    return G


def load_one_saved_split(
    split_path: Path | str,
    num_nodes: int,
) -> Dict[str, np.ndarray | nx.Graph]:
    split_path = Path(split_path)
    data = np.load(split_path, allow_pickle=True)

    train_graph_edges = data["train_graph_edges"].astype(np.int64)
    train_graph = reconstruct_train_graph(train_graph_edges, num_nodes)

    split_dict: Dict[str, np.ndarray | nx.Graph] = {
        "train_graph": train_graph,
        "train_pos": data["train_pos"].astype(np.int64),
        "val_pos": data["val_pos"].astype(np.int64),
        "test_pos": data["test_pos"].astype(np.int64),
        "train_neg": data["train_neg"].astype(np.int64),
        "val_neg": data["val_neg"].astype(np.int64),
        "test_neg": data["test_neg"].astype(np.int64),
    }
    return split_dict


def load_all_saved_splits(
    graphs: Dict[str, nx.Graph],
    split_dir: Path | str = "artifacts/splits",
) -> Dict[str, Dict[str, np.ndarray | nx.Graph]]:
    split_dir = Path(split_dir)

    all_splits: Dict[str, Dict[str, np.ndarray | nx.Graph]] = {}

    for name, G in graphs.items():
        split_path = split_dir / f"{name.lower()}_split.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        split_dict = load_one_saved_split(split_path, num_nodes=G.number_of_nodes())
        all_splits[name] = split_dict

    return all_splits


def build_neighbor_sets(train_graph: nx.Graph) -> List[set[int]]:
    n = train_graph.number_of_nodes()
    neighbors: List[set[int]] = [set() for _ in range(n)]

    for u in range(n):
        neighbors[u] = set(train_graph.neighbors(u))

    return neighbors


def build_degree_array(train_graph: nx.Graph) -> np.ndarray:
    n = train_graph.number_of_nodes()
    deg = np.zeros(n, dtype=np.int64)

    for node, d in train_graph.degree():
        deg[node] = d

    return deg


def build_adamic_adar_weights(deg: np.ndarray) -> np.ndarray:
    aa_weight = np.zeros_like(deg, dtype=np.float32)

    valid = deg > 1
    aa_weight[valid] = 1.0 / np.log(deg[valid].astype(np.float64))

    return aa_weight.astype(np.float32)


def score_common_neighbors(
    edge_pairs: np.ndarray,
    neighbors: List[set[int]],
) -> np.ndarray:
    scores = np.zeros(len(edge_pairs), dtype=np.float32)

    for i, (u, v) in enumerate(edge_pairs):
        scores[i] = float(len(neighbors[u] & neighbors[v]))

    return scores


def score_jaccard(
    edge_pairs: np.ndarray,
    neighbors: List[set[int]],
) -> np.ndarray:
    scores = np.zeros(len(edge_pairs), dtype=np.float32)

    for i, (u, v) in enumerate(edge_pairs):
        nu = neighbors[u]
        nv = neighbors[v]

        inter = len(nu & nv)
        union = len(nu | nv)

        scores[i] = 0.0 if union == 0 else float(inter / union)

    return scores


def score_adamic_adar(
    edge_pairs: np.ndarray,
    neighbors: List[set[int]],
    aa_weight: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(edge_pairs), dtype=np.float32)

    for i, (u, v) in enumerate(edge_pairs):
        common = neighbors[u] & neighbors[v]
        if common:
            scores[i] = float(np.sum(aa_weight[list(common)]))
        else:
            scores[i] = 0.0

    return scores


def score_preferential_attachment(
    edge_pairs: np.ndarray,
    deg: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(edge_pairs), dtype=np.float32)

    for i, (u, v) in enumerate(edge_pairs):
        scores[i] = float(deg[u] * deg[v])

    return scores


def build_labels_and_scores(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.int64),
            np.zeros(len(neg_scores), dtype=np.int64),
        ]
    )
    y_score = np.concatenate([pos_scores, neg_scores]).astype(np.float32)
    return y_true, y_score


def evaluate_scores(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> Dict[str, float]:
    y_true, y_score = build_labels_and_scores(pos_scores, neg_scores)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auc": float(auc),
        "ap": float(ap),
    }


def evaluate_one_method(
    method_name: str,
    train_graph: nx.Graph,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    neighbors: List[set[int]],
    deg: np.ndarray,
    aa_weight: np.ndarray,
) -> Dict[str, float]:
    if method_name == "common_neighbors":
        pos_scores = score_common_neighbors(pos_edges, neighbors)
        neg_scores = score_common_neighbors(neg_edges, neighbors)

    elif method_name == "jaccard":
        pos_scores = score_jaccard(pos_edges, neighbors)
        neg_scores = score_jaccard(neg_edges, neighbors)

    elif method_name == "adamic_adar":
        pos_scores = score_adamic_adar(pos_edges, neighbors, aa_weight)
        neg_scores = score_adamic_adar(neg_edges, neighbors, aa_weight)

    elif method_name == "preferential_attachment":
        pos_scores = score_preferential_attachment(pos_edges, deg)
        neg_scores = score_preferential_attachment(neg_edges, deg)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics = evaluate_scores(pos_scores, neg_scores)
    return metrics


def evaluate_baselines_for_graph(
    graph_name: str,
    split_dict: Dict[str, np.ndarray | nx.Graph],
) -> List[Dict[str, object]]:
    train_graph = split_dict["train_graph"]
    val_pos = split_dict["val_pos"]
    val_neg = split_dict["val_neg"]
    test_pos = split_dict["test_pos"]
    test_neg = split_dict["test_neg"]

    if not isinstance(train_graph, nx.Graph):
        raise TypeError("split_dict['train_graph'] must be a NetworkX graph.")

    neighbors = build_neighbor_sets(train_graph)
    deg = build_degree_array(train_graph)
    aa_weight = build_adamic_adar_weights(deg)

    methods = [
        "common_neighbors",
        "jaccard",
        "adamic_adar",
        "preferential_attachment",
    ]

    results: List[Dict[str, object]] = []

    for method in methods:
        val_metrics = evaluate_one_method(
            method_name=method,
            train_graph=train_graph,
            pos_edges=val_pos,
            neg_edges=val_neg,
            neighbors=neighbors,
            deg=deg,
            aa_weight=aa_weight,
        )

        test_metrics = evaluate_one_method(
            method_name=method,
            train_graph=train_graph,
            pos_edges=test_pos,
            neg_edges=test_neg,
            neighbors=neighbors,
            deg=deg,
            aa_weight=aa_weight,
        )

        results.append(
            {
                "graph": graph_name,
                "method": method,
                "val_auc": val_metrics["auc"],
                "val_ap": val_metrics["ap"],
                "test_auc": test_metrics["auc"],
                "test_ap": test_metrics["ap"],
            }
        )

    return results


def print_results_table(results: List[Dict[str, object]]) -> None:
    print(
        f"\n{'Graph':<12} {'Method':<28} "
        f"{'Val AUC':>9} {'Val AP':>9} {'Test AUC':>10} {'Test AP':>9}"
    )
    print("-" * 82)

    for row in results:
        print(
            f"{row['graph']:<12} "
            f"{row['method']:<28} "
            f"{row['val_auc']:>9.4f} "
            f"{row['val_ap']:>9.4f} "
            f"{row['test_auc']:>10.4f} "
            f"{row['test_ap']:>9.4f}"
        )


def save_results_csv(
    results: List[Dict[str, object]],
    save_path: Path | str = "artifacts/baselines/baseline_results.csv",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["graph", "method", "val_auc", "val_ap", "test_auc", "test_ap"]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved baseline results to: {save_path}")


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=True, relabel=True)
    all_splits = load_all_saved_splits(graphs, split_dir="artifacts/splits")

    all_results: List[Dict[str, object]] = []

    for graph_name in ["Facebook", "Enron", "Erdos"]:
        print(f"\nEvaluating baselines on {graph_name}...")
        results = evaluate_baselines_for_graph(graph_name, all_splits[graph_name])
        all_results.extend(results)

    print_results_table(all_results)
    save_results_csv(all_results, save_path="artifacts/baselines/baseline_results.csv")


if __name__ == "__main__":
    main()
