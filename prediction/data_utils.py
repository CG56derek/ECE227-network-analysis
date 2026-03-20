from __future__ import annotations

from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
from scipy import io, sparse


def resolve_data_dir(explicit_data_dir: Path | str | None = None) -> Path:
    candidates: list[Path] = []

    if explicit_data_dir is not None:
        candidates.append(Path(explicit_data_dir))

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates.extend(
        [
            script_dir / "data",
            script_dir.parent / "data",
            cwd / "data",
            cwd,
        ]
    )

    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find a valid data directory. Searched:\n{searched}")


def resolve_data_file(data_dir: Path | str, candidates: list[str]) -> Path:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")


    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path.resolve()


    files = [p for p in data_dir.iterdir() if p.is_file()]
    lowered_candidates = {c.lower() for c in candidates}
    lowered_stems = {Path(c).stem.lower() for c in candidates}

    for p in files:
        if p.name.lower() in lowered_candidates:
            return p.resolve()
        if p.stem.lower() in lowered_stems:
            return p.resolve()

    available = "\n".join(p.name for p in files) if files else "(no files found)"
    raise FileNotFoundError(
        f"Could not find any of these files in {data_dir}:\n"
        + "\n".join(candidates)
        + f"\n\nAvailable files:\n{available}"
    )


def preprocess_graph(
    G: nx.Graph,
    use_gcc: bool = True,
    relabel: bool = True,
) -> nx.Graph:
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    if use_gcc:
        if G.number_of_nodes() == 0:
            raise ValueError("Graph is empty after cleaning.")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    if relabel:
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    return G


def graph_stats(G: nx.Graph, name: str = "Graph") -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    avg_degree = (2.0 * m / n) if n > 0 else 0.0
    num_components = nx.number_connected_components(G)

    return {
        "name": name,
        "nodes": n,
        "edges": m,
        "components": num_components,
        "density": density,
        "avg_degree": avg_degree,
    }


def print_graph_stats(G: nx.Graph, name: str = "Graph") -> None:
    stats = graph_stats(G, name)
    print(f"\n[{stats['name']}]")
    print(f"nodes       = {stats['nodes']}")
    print(f"edges       = {stats['edges']}")
    print(f"components  = {stats['components']}")
    print(f"density     = {stats['density']:.6f}")
    print(f"avg_degree  = {stats['avg_degree']:.4f}")


def load_edge_list_graph(
    file_path: Path | str,
    nodetype=int,
    comments: str = "#",
) -> nx.Graph:
    file_path = Path(file_path)
    G = nx.read_edgelist(
        file_path,
        comments=comments,
        nodetype=nodetype,
        create_using=nx.Graph(),
    )
    return G


def _extract_matrix_from_mat(mat_dict: dict):
    common_keys = ["A", "adjacency", "Adj", "Network", "network"]

    for key in common_keys:
        if key in mat_dict:
            return mat_dict[key]

    if "Problem" in mat_dict:
        problem = mat_dict["Problem"]


        if hasattr(problem, "A"):
            return problem.A


        if isinstance(problem, np.ndarray) and problem.dtype.names is not None:
            if "A" in problem.dtype.names:
                return problem["A"][()]


        try:
            return problem[0, 0]["A"]
        except Exception:
            pass

        try:
            return problem[0, 0].A
        except Exception:
            pass

    valid_keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    raise KeyError(
        "Could not find adjacency matrix in .mat file. "
        f"Available keys: {valid_keys}"
    )


def load_mat_graph(file_path: Path | str) -> nx.Graph:
    file_path = Path(file_path)

    mat_dict = io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    matrix = _extract_matrix_from_mat(mat_dict)

    if sparse.issparse(matrix):
        G = nx.from_scipy_sparse_array(matrix)
    else:
        matrix = np.asarray(matrix)
        G = nx.from_numpy_array(matrix)

    return G


def load_facebook_graph(
    data_dir: Path | str,
    use_gcc: bool = True,
    relabel: bool = True,
) -> nx.Graph:
    file_path = resolve_data_file(
        data_dir,
        ["facebook_combined.txt", "facebook_combined"],
    )
    print(f"Loading Facebook from: {file_path}")
    G = load_edge_list_graph(file_path)
    G = preprocess_graph(G, use_gcc=use_gcc, relabel=relabel)
    return G


def load_enron_graph(
    data_dir: Path | str,
    use_gcc: bool = True,
    relabel: bool = True,
) -> nx.Graph:
    file_path = resolve_data_file(
        data_dir,
        ["email-Enron.txt", "email-Enron"],
    )
    print(f"Loading Enron from: {file_path}")
    G = load_edge_list_graph(file_path)
    G = preprocess_graph(G, use_gcc=use_gcc, relabel=relabel)
    return G


def load_erdos_graph(
    data_dir: Path | str,
    use_gcc: bool = True,
    relabel: bool = True,
) -> nx.Graph:
    file_path = resolve_data_file(
        data_dir,
        ["Erdos02.mat", "erdos02.mat", "Erdos02"],
    )
    print(f"Loading Erdos from: {file_path}")
    G = load_mat_graph(file_path)
    G = preprocess_graph(G, use_gcc=use_gcc, relabel=relabel)
    return G


def load_all_graphs(
    data_dir: Path | str,
    use_gcc: bool = True,
    relabel: bool = True,
) -> Dict[str, nx.Graph]:
    graphs = {
        "Facebook": load_facebook_graph(data_dir, use_gcc=use_gcc, relabel=relabel),
        "Enron": load_enron_graph(data_dir, use_gcc=use_gcc, relabel=relabel),
        "Erdos": load_erdos_graph(data_dir, use_gcc=use_gcc, relabel=relabel),
    }
    return graphs


def main() -> None:
    data_dir = resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    graphs = load_all_graphs(data_dir=data_dir, use_gcc=True, relabel=True)

    for name, G in graphs.items():
        print_graph_stats(G, name)


if __name__ == "__main__":
    main()
