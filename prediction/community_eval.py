from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data_utils import load_all_graphs, resolve_data_dir
from features import build_feature_matrix
from model import GraphSAGELinkPredictor


def run_louvain_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> List[set[int]]:
    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(
            G,
            resolution=resolution,
            seed=seed,
        )
        return [set(c) for c in communities]

    except Exception:
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(
                G,
                resolution=resolution,
                random_state=seed,
            )

            comm_to_nodes: Dict[int, set[int]] = {}
            for node, comm_id in partition.items():
                comm_to_nodes.setdefault(comm_id, set()).add(node)

            communities = list(comm_to_nodes.values())
            return communities

        except Exception as e:
            raise ImportError(
                "Could not run Louvain community detection. "
                "Please use a recent NetworkX version or install python-louvain."
            ) from e


def build_node_to_community(communities: List[set[int]]) -> Dict[int, int]:
    node_to_comm: Dict[int, int] = {}

    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            node_to_comm[node] = comm_id

    return node_to_comm


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

    return {
        "train_graph": train_graph,
        "train_pos": data["train_pos"].astype(np.int64),
        "val_pos": data["val_pos"].astype(np.int64),
        "test_pos": data["test_pos"].astype(np.int64),
        "train_neg": data["train_neg"].astype(np.int64),
        "val_neg": data["val_neg"].astype(np.int64),
        "test_neg": data["test_neg"].astype(np.int64),
    }


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


def build_train_features(train_graph: nx.Graph) -> torch.Tensor:
    X, _, _ = build_feature_matrix(train_graph, standardize=True)
    return torch.tensor(X, dtype=torch.float32)


def build_edge_index_from_nx(train_graph: nx.Graph) -> torch.Tensor:
    edge_list: List[List[int]] = []

    for u, v in train_graph.edges():
        edge_list.append([u, v])
        edge_list.append([v, u])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def edge_array_to_tensor(edge_array: np.ndarray) -> torch.Tensor:
    return torch.tensor(edge_array, dtype=torch.long)


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


def evaluate_scores(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> Dict[str, float]:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {"auc": float("nan"), "ap": float("nan")}

    y_true = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.int64),
            np.zeros(len(neg_scores), dtype=np.int64),
        ]
    )
    y_score = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auc": float(auc),
        "ap": float(ap),
    }


def partition_edges_by_community(
    edge_pairs: np.ndarray,
    node_to_comm: Dict[int, int],
) -> Dict[str, np.ndarray]:
    intra_edges: List[List[int]] = []
    inter_edges: List[List[int]] = []

    for u, v in edge_pairs:
        if node_to_comm[u] == node_to_comm[v]:
            intra_edges.append([int(u), int(v)])
        else:
            inter_edges.append([int(u), int(v)])

    intra_array = np.array(intra_edges, dtype=np.int64).reshape(-1, 2)
    inter_array = np.array(inter_edges, dtype=np.int64).reshape(-1, 2)

    return {
        "intra": intra_array,
        "inter": inter_array,
    }


def load_graphsage_model(
    graph_name: str,
    in_channels: int,
    hidden_channels: int = 32,
    dropout: float = 0.3,
    device: str | torch.device = "cpu",
    model_dir: Path | str = "artifacts/models",
) -> GraphSAGELinkPredictor:
    device = torch.device(device)
    model_dir = Path(model_dir)
    model_path = model_dir / f"graphsage_{graph_name.lower()}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = GraphSAGELinkPredictor(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def score_graphsage(
    z: torch.Tensor,
    model: GraphSAGELinkPredictor,
    edge_pairs: np.ndarray,
    device: str | torch.device,
) -> np.ndarray:
    if len(edge_pairs) == 0:
        return np.array([], dtype=np.float32)

    device = torch.device(device)
    edge_tensor = edge_array_to_tensor(edge_pairs).to(device)
    probs = model.decode_proba(z, edge_tensor).cpu().numpy().astype(np.float32)
    return probs


def evaluate_one_graph_community_cases(
    graph_name: str,
    split_dict: Dict[str, np.ndarray | nx.Graph],
    device: str | torch.device = "cpu",
    louvain_resolution: float = 1.0,
    louvain_seed: int = 42,
) -> List[Dict[str, object]]:
    train_graph = split_dict["train_graph"]
    test_pos = split_dict["test_pos"]
    test_neg = split_dict["test_neg"]

    if not isinstance(train_graph, nx.Graph):
        raise TypeError("split_dict['train_graph'] must be a NetworkX graph.")


    communities = run_louvain_communities(
        train_graph,
        resolution=louvain_resolution,
        seed=louvain_seed,
    )
    node_to_comm = build_node_to_community(communities)

    pos_parts = partition_edges_by_community(test_pos, node_to_comm)
    neg_parts = partition_edges_by_community(test_neg, node_to_comm)


    neighbors = build_neighbor_sets(train_graph)
    deg = build_degree_array(train_graph)
    aa_weight = build_adamic_adar_weights(deg)


    device = torch.device(device)
    x = build_train_features(train_graph).to(device)
    edge_index = build_edge_index_from_nx(train_graph).to(device)

    model = load_graphsage_model(
        graph_name=graph_name,
        in_channels=x.size(1),
        hidden_channels=32,
        dropout=0.3,
        device=device,
        model_dir="artifacts/models",
    )

    with torch.no_grad():
        z = model.encode(x, edge_index)

    results: List[Dict[str, object]] = []

    print(f"\n[{graph_name} community evaluation]")
    print(f"num_communities = {len(communities)}")
    print(f"intra test positives = {len(pos_parts['intra'])}")
    print(f"inter test positives = {len(pos_parts['inter'])}")
    print(f"intra test negatives = {len(neg_parts['intra'])}")
    print(f"inter test negatives = {len(neg_parts['inter'])}")

    for case_name in ["intra", "inter"]:
        pos_edges = pos_parts[case_name]
        neg_edges = neg_parts[case_name]


        aa_pos = score_adamic_adar(pos_edges, neighbors, aa_weight)
        aa_neg = score_adamic_adar(neg_edges, neighbors, aa_weight)
        aa_metrics = evaluate_scores(aa_pos, aa_neg)

        results.append(
            {
                "graph": graph_name,
                "community_case": case_name,
                "method": "Adamic-Adar",
                "num_communities": len(communities),
                "num_pos_edges": int(len(pos_edges)),
                "num_neg_edges": int(len(neg_edges)),
                "test_auc": aa_metrics["auc"],
                "test_ap": aa_metrics["ap"],
            }
        )


        pa_pos = score_preferential_attachment(pos_edges, deg)
        pa_neg = score_preferential_attachment(neg_edges, deg)
        pa_metrics = evaluate_scores(pa_pos, pa_neg)

        results.append(
            {
                "graph": graph_name,
                "community_case": case_name,
                "method": "Preferential Attachment",
                "num_communities": len(communities),
                "num_pos_edges": int(len(pos_edges)),
                "num_neg_edges": int(len(neg_edges)),
                "test_auc": pa_metrics["auc"],
                "test_ap": pa_metrics["ap"],
            }
        )


        gs_pos = score_graphsage(z, model, pos_edges, device)
        gs_neg = score_graphsage(z, model, neg_edges, device)
        gs_metrics = evaluate_scores(gs_pos, gs_neg)

        results.append(
            {
                "graph": graph_name,
                "community_case": case_name,
                "method": "GraphSAGE",
                "num_communities": len(communities),
                "num_pos_edges": int(len(pos_edges)),
                "num_neg_edges": int(len(neg_edges)),
                "test_auc": gs_metrics["auc"],
                "test_ap": gs_metrics["ap"],
            }
        )

    return results


def print_results_table(results: List[Dict[str, object]]) -> None:
    print(
        f"\n{'Graph':<12} {'Case':<10} {'Method':<28} "
        f"{'Pos':>7} {'Neg':>7} {'Test AUC':>10} {'Test AP':>9}"
    )
    print("-" * 92)

    for row in results:
        print(
            f"{str(row['graph']):<12} "
            f"{str(row['community_case']):<10} "
            f"{str(row['method']):<28} "
            f"{int(row['num_pos_edges']):>7d} "
            f"{int(row['num_neg_edges']):>7d} "
            f"{float(row['test_auc']):>10.4f} "
            f"{float(row['test_ap']):>9.4f}"
        )


def save_results_csv(
    results: List[Dict[str, object]],
    save_path: Path | str = "artifacts/community/community_eval_results.csv",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved community-aware results to: {save_path}")


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=True, relabel=True)
    all_splits = load_all_saved_splits(graphs, split_dir="artifacts/splits")

    all_results: List[Dict[str, object]] = []

    for graph_name in ["Facebook", "Enron", "Erdos"]:
        results = evaluate_one_graph_community_cases(
            graph_name=graph_name,
            split_dict=all_splits[graph_name],
            device=device,
            louvain_resolution=1.0,
            louvain_seed=42,
        )
        all_results.extend(results)

    print_results_table(all_results)
    save_results_csv(
        all_results,
        save_path="artifacts/community/community_eval_results.csv",
    )


if __name__ == "__main__":
    main()
