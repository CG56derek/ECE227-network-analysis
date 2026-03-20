from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List
import csv
import random

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data_utils import load_all_graphs, resolve_data_dir
from features import build_feature_matrix
from model import GraphSAGELinkPredictor, build_bce_loss, count_parameters


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    train_graph = reconstruct_train_graph(train_graph_edges, num_nodes=num_nodes)

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


def build_edge_index_from_nx(train_graph: nx.Graph) -> torch.Tensor:
    edge_list: List[List[int]] = []

    for u, v in train_graph.edges():
        edge_list.append([u, v])
        edge_list.append([v, u])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def edge_array_to_tensor(edge_array: np.ndarray) -> torch.Tensor:
    return torch.tensor(edge_array, dtype=torch.long)


def build_train_features(train_graph: nx.Graph) -> torch.Tensor:
    X, _, _ = build_feature_matrix(train_graph, standardize=True)
    x = torch.tensor(X, dtype=torch.float32)
    return x


def evaluate_edge_scores(
    model: GraphSAGELinkPredictor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        z = model.encode(x, edge_index)

        pos_logits = model.decode(z, pos_edges)
        neg_logits = model.decode(z, neg_edges)

        pos_probs = torch.sigmoid(pos_logits).cpu().numpy()
        neg_probs = torch.sigmoid(neg_logits).cpu().numpy()

    y_true = np.concatenate(
        [
            np.ones(len(pos_probs), dtype=np.int64),
            np.zeros(len(neg_probs), dtype=np.int64),
        ]
    )
    y_score = np.concatenate([pos_probs, neg_probs])

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auc": float(auc),
        "ap": float(ap),
    }


def train_one_epoch(
    model: GraphSAGELinkPredictor,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    train_pos: torch.Tensor,
    train_neg: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, edge_index)

    pos_logits = model.decode(z, train_pos)
    neg_logits = model.decode(z, train_neg)

    loss = build_bce_loss(pos_logits, neg_logits)
    loss.backward()
    optimizer.step()

    return float(loss.item())


def save_history_csv(
    history: List[Dict[str, float | int]],
    save_path: Path | str,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def train_graphsage_for_one_graph(
    graph_name: str,
    split_dict: Dict[str, np.ndarray | nx.Graph],
    hidden_channels: int = 32,
    dropout: float = 0.3,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
    device: str | torch.device = "cpu",
    model_dir: Path | str = "artifacts/models",
    history_dir: Path | str = "artifacts/training_history",
) -> Dict[str, float | int | str]:
    set_seed(seed)

    train_graph = split_dict["train_graph"]
    train_pos_np = split_dict["train_pos"]
    val_pos_np = split_dict["val_pos"]
    test_pos_np = split_dict["test_pos"]
    train_neg_np = split_dict["train_neg"]
    val_neg_np = split_dict["val_neg"]
    test_neg_np = split_dict["test_neg"]

    if not isinstance(train_graph, nx.Graph):
        raise TypeError("split_dict['train_graph'] must be a NetworkX graph.")

    device = torch.device(device)


    x = build_train_features(train_graph).to(device)
    edge_index = build_edge_index_from_nx(train_graph).to(device)

    train_pos = edge_array_to_tensor(train_pos_np).to(device)
    val_pos = edge_array_to_tensor(val_pos_np).to(device)
    test_pos = edge_array_to_tensor(test_pos_np).to(device)

    train_neg = edge_array_to_tensor(train_neg_np).to(device)
    val_neg = edge_array_to_tensor(val_neg_np).to(device)
    test_neg = edge_array_to_tensor(test_neg_np).to(device)

    model = GraphSAGELinkPredictor(
        in_channels=x.size(1),
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_params = count_parameters(model)

    best_val_ap = -float("inf")
    best_epoch = -1
    best_state = None
    best_test_auc = None
    best_test_ap = None
    best_val_auc = None

    epochs_without_improvement = 0
    history: List[Dict[str, float | int]] = []

    print(f"\nTraining GraphSAGE on {graph_name}...")
    print(f"device         = {device}")
    print(f"num_nodes      = {train_graph.number_of_nodes()}")
    print(f"train_edges    = {train_graph.number_of_edges()}")
    print(f"input_dim      = {x.size(1)}")
    print(f"hidden_dim     = {hidden_channels}")
    print(f"num_params     = {num_params}")

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            x=x,
            edge_index=edge_index,
            train_pos=train_pos,
            train_neg=train_neg,
        )

        val_metrics = evaluate_edge_scores(
            model=model,
            x=x,
            edge_index=edge_index,
            pos_edges=val_pos,
            neg_edges=val_neg,
        )

        test_metrics = evaluate_edge_scores(
            model=model,
            x=x,
            edge_index=edge_index,
            pos_edges=test_pos,
            neg_edges=test_neg,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(loss),
            "val_auc": float(val_metrics["auc"]),
            "val_ap": float(val_metrics["ap"]),
            "test_auc": float(test_metrics["auc"]),
            "test_ap": float(test_metrics["ap"]),
        }
        history.append(row)

        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss:.4f} "
                f"val_auc={val_metrics['auc']:.4f} "
                f"val_ap={val_metrics['ap']:.4f} "
                f"test_auc={test_metrics['auc']:.4f} "
                f"test_ap={test_metrics['ap']:.4f}"
            )


        if val_metrics["ap"] > best_val_ap:
            best_val_ap = float(val_metrics["ap"])
            best_val_auc = float(val_metrics["auc"])
            best_test_auc = float(test_metrics["auc"])
            best_test_ap = float(test_metrics["ap"])
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model was saved.")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"graphsage_{graph_name.lower()}.pt"
    torch.save(best_state, model_path)

    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"graphsage_{graph_name.lower()}_history.csv"
    save_history_csv(history, history_path)

    print(
        f"Best {graph_name}: "
        f"epoch={best_epoch}, "
        f"val_auc={best_val_auc:.4f}, "
        f"val_ap={best_val_ap:.4f}, "
        f"test_auc={best_test_auc:.4f}, "
        f"test_ap={best_test_ap:.4f}"
    )
    print(f"Saved model   to: {model_path}")
    print(f"Saved history to: {history_path}")

    return {
        "graph": graph_name,
        "model": "GraphSAGE",
        "best_epoch": int(best_epoch),
        "num_params": int(num_params),
        "val_auc": float(best_val_auc),
        "val_ap": float(best_val_ap),
        "test_auc": float(best_test_auc),
        "test_ap": float(best_test_ap),
    }


def save_results_csv(
    results: List[Dict[str, float | int | str]],
    save_path: Path | str = "artifacts/results/graphsage_results.csv",
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

    print(f"\nSaved GraphSAGE results to: {save_path}")


def print_results_table(results: List[Dict[str, float | int | str]]) -> None:
    print(
        f"\n{'Graph':<12} {'Model':<12} {'Best Ep':>8} "
        f"{'Val AUC':>9} {'Val AP':>9} {'Test AUC':>10} {'Test AP':>9}"
    )
    print("-" * 78)

    for row in results:
        print(
            f"{str(row['graph']):<12} "
            f"{str(row['model']):<12} "
            f"{int(row['best_epoch']):>8d} "
            f"{float(row['val_auc']):>9.4f} "
            f"{float(row['val_ap']):>9.4f} "
            f"{float(row['test_auc']):>10.4f} "
            f"{float(row['test_ap']):>9.4f}"
        )


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=True, relabel=True)
    all_splits = load_all_saved_splits(graphs, split_dir="artifacts/splits")

    results: List[Dict[str, float | int | str]] = []

    for graph_name in ["Facebook", "Enron", "Erdos"]:
        result = train_graphsage_for_one_graph(
            graph_name=graph_name,
            split_dict=all_splits[graph_name],
            hidden_channels=32,
            dropout=0.3,
            lr=0.01,
            weight_decay=5e-4,
            max_epochs=100,
            patience=20,
            seed=42,
            device=device,
            model_dir="artifacts/models",
            history_dir="artifacts/training_history",
        )
        results.append(result)

    print_results_table(results)
    save_results_csv(results, save_path="artifacts/results/graphsage_results.csv")


if __name__ == "__main__":
    main()
