from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from data_utils import load_all_graphs, resolve_data_dir


def check_contiguous_node_labels(G: nx.Graph) -> None:
    nodes = sorted(G.nodes())
    expected = list(range(G.number_of_nodes()))
    if nodes != expected:
        raise ValueError(
            "Node labels are not contiguous 0..N-1. "
            "Please relabel the graph before splitting."
        )


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def edge_array_from_graph(G: nx.Graph) -> np.ndarray:
    edges = [canonical_edge(u, v) for u, v in G.edges()]
    edges = np.array(sorted(edges), dtype=np.int64)
    return edges


def sample_negative_edges(
    G: nx.Graph,
    num_samples: int,
    rng: np.random.Generator,
    forbidden_edges: set[Tuple[int, int]] | None = None,
) -> np.ndarray:
    n = G.number_of_nodes()
    existing_edges = {canonical_edge(u, v) for u, v in G.edges()}

    if forbidden_edges is None:
        forbidden_edges = set()

    negatives: set[Tuple[int, int]] = set()

    while len(negatives) < num_samples:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))

        if u == v:
            continue

        e = canonical_edge(u, v)

        if e in existing_edges:
            continue
        if e in forbidden_edges:
            continue
        if e in negatives:
            continue

        negatives.add(e)

    neg_array = np.array(sorted(negatives), dtype=np.int64)
    return neg_array


def build_connected_edge_split(
    G: nx.Graph,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, np.ndarray | nx.Graph]:
    check_contiguous_node_labels(G)

    if not nx.is_connected(G):
        raise ValueError("Input graph must be connected. Please pass the GCC.")

    rng = np.random.default_rng(seed)

    all_edges = {canonical_edge(u, v) for u, v in G.edges()}
    num_edges = len(all_edges)

    num_val = int(round(num_edges * val_ratio))
    num_test = int(round(num_edges * test_ratio))
    num_holdout = num_val + num_test


    T = nx.minimum_spanning_tree(G)
    tree_edges = {canonical_edge(u, v) for u, v in T.edges()}

    non_tree_edges = list(all_edges - tree_edges)

    if len(non_tree_edges) < num_holdout:
        raise ValueError(
            "Not enough non-tree edges to create validation/test splits "
            f"while keeping training graph connected.\n"
            f"non_tree_edges={len(non_tree_edges)}, required={num_holdout}"
        )

    rng.shuffle(non_tree_edges)

    val_pos = np.array(sorted(non_tree_edges[:num_val]), dtype=np.int64)
    test_pos = np.array(sorted(non_tree_edges[num_val:num_val + num_test]), dtype=np.int64)

    held_out = set(map(tuple, val_pos)) | set(map(tuple, test_pos))
    train_pos_set = all_edges - held_out
    train_pos = np.array(sorted(train_pos_set), dtype=np.int64)


    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_pos.tolist())

    if not nx.is_connected(train_graph):
        raise RuntimeError("Training graph is not connected. This should not happen.")


    used_negative_edges: set[Tuple[int, int]] = set()

    train_neg = sample_negative_edges(
        G=G,
        num_samples=len(train_pos),
        rng=rng,
        forbidden_edges=used_negative_edges,
    )
    used_negative_edges |= set(map(tuple, train_neg))

    val_neg = sample_negative_edges(
        G=G,
        num_samples=len(val_pos),
        rng=rng,
        forbidden_edges=used_negative_edges,
    )
    used_negative_edges |= set(map(tuple, val_neg))

    test_neg = sample_negative_edges(
        G=G,
        num_samples=len(test_pos),
        rng=rng,
        forbidden_edges=used_negative_edges,
    )
    used_negative_edges |= set(map(tuple, test_neg))

    return {
        "train_graph": train_graph,
        "train_pos": train_pos,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "train_neg": train_neg,
        "val_neg": val_neg,
        "test_neg": test_neg,
    }


def sanity_check_split(
    G: nx.Graph,
    split_dict: Dict[str, np.ndarray | nx.Graph],
) -> None:
    train_graph = split_dict["train_graph"]
    train_pos = split_dict["train_pos"]
    val_pos = split_dict["val_pos"]
    test_pos = split_dict["test_pos"]
    train_neg = split_dict["train_neg"]
    val_neg = split_dict["val_neg"]
    test_neg = split_dict["test_neg"]

    all_pos = set(map(tuple, train_pos)) | set(map(tuple, val_pos)) | set(map(tuple, test_pos))
    original_edges = {canonical_edge(u, v) for u, v in G.edges()}

    if all_pos != original_edges:
        raise AssertionError("Positive splits do not reconstruct the original edge set.")


    if set(map(tuple, train_pos)) & set(map(tuple, val_pos)):
        raise AssertionError("train_pos and val_pos overlap.")
    if set(map(tuple, train_pos)) & set(map(tuple, test_pos)):
        raise AssertionError("train_pos and test_pos overlap.")
    if set(map(tuple, val_pos)) & set(map(tuple, test_pos)):
        raise AssertionError("val_pos and test_pos overlap.")


    if set(map(tuple, train_neg)) & set(map(tuple, val_neg)):
        raise AssertionError("train_neg and val_neg overlap.")
    if set(map(tuple, train_neg)) & set(map(tuple, test_neg)):
        raise AssertionError("train_neg and test_neg overlap.")
    if set(map(tuple, val_neg)) & set(map(tuple, test_neg)):
        raise AssertionError("val_neg and test_neg overlap.")


    for name, neg_edges in [("train_neg", train_neg), ("val_neg", val_neg), ("test_neg", test_neg)]:
        for e in map(tuple, neg_edges):
            if e in original_edges:
                raise AssertionError(f"{name} contains a real edge: {e}")


    train_graph_edges = {canonical_edge(u, v) for u, v in train_graph.edges()}
    if train_graph_edges != set(map(tuple, train_pos)):
        raise AssertionError("train_graph edges do not match train_pos.")

    if not nx.is_connected(train_graph):
        raise AssertionError("train_graph is not connected.")


def summarize_split(
    graph_name: str,
    G: nx.Graph,
    split_dict: Dict[str, np.ndarray | nx.Graph],
) -> None:
    train_graph = split_dict["train_graph"]
    train_pos = split_dict["train_pos"]
    val_pos = split_dict["val_pos"]
    test_pos = split_dict["test_pos"]
    train_neg = split_dict["train_neg"]
    val_neg = split_dict["val_neg"]
    test_neg = split_dict["test_neg"]

    print(f"\n[{graph_name} split summary]")
    print(f"original graph : nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    print(
        f"train graph    : nodes={train_graph.number_of_nodes()}, "
        f"edges={train_graph.number_of_edges()}, "
        f"connected={nx.is_connected(train_graph)}"
    )
    print(f"train_pos      : {len(train_pos)}")
    print(f"val_pos        : {len(val_pos)}")
    print(f"test_pos       : {len(test_pos)}")
    print(f"train_neg      : {len(train_neg)}")
    print(f"val_neg        : {len(val_neg)}")
    print(f"test_neg       : {len(test_neg)}")


def save_one_split(
    save_path: Path | str,
    split_dict: Dict[str, np.ndarray | nx.Graph],
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_graph = split_dict["train_graph"]
    train_graph_edges = edge_array_from_graph(train_graph)

    np.savez_compressed(
        save_path,
        train_graph_edges=train_graph_edges,
        train_pos=split_dict["train_pos"],
        val_pos=split_dict["val_pos"],
        test_pos=split_dict["test_pos"],
        train_neg=split_dict["train_neg"],
        val_neg=split_dict["val_neg"],
        test_neg=split_dict["test_neg"],
    )


def save_all_splits(
    all_splits: Dict[str, Dict[str, np.ndarray | nx.Graph]],
    out_dir: Path | str = "artifacts/splits",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_dict in all_splits.items():
        save_path = out_dir / f"{name.lower()}_split.npz"
        save_one_split(save_path, split_dict)
        print(f"Saved {name} split to: {save_path}")


def build_all_splits(
    data_dir: Path | str | None = None,
    use_gcc: bool = True,
    relabel: bool = True,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray | nx.Graph]]:
    if data_dir is None:
        data_dir = resolve_data_dir()

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=use_gcc, relabel=relabel)

    all_splits: Dict[str, Dict[str, np.ndarray | nx.Graph]] = {}

    for name, G in graphs.items():
        split_dict = build_connected_edge_split(
            G=G,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        sanity_check_split(G, split_dict)
        all_splits[name] = split_dict

    return all_splits


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=True, relabel=True)

    all_splits: Dict[str, Dict[str, np.ndarray | nx.Graph]] = {}

    for name, G in graphs.items():
        split_dict = build_connected_edge_split(
            G=G,
            val_ratio=0.10,
            test_ratio=0.10,
            seed=42,
        )

        sanity_check_split(G, split_dict)
        summarize_split(name, G, split_dict)
        all_splits[name] = split_dict

    save_all_splits(all_splits, out_dir="artifacts/splits")


if __name__ == "__main__":
    main()
