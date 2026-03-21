import argparse
import gzip
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import loadmat


DEFAULT_FILES = {
    "facebook": "data/facebook_combined.txt.gz",
    "enron": "data/email-Enron.txt.gz",
    "erdos992": "data/Erdos02.mat",
}

DEFAULT_SEED = 42


def load_snap_edgelist_gz(path: Path) -> nx.Graph:
    G = nx.Graph()
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = line.split()
            G.add_edge(int(u), int(v))
    return G


def _mat_char_matrix_to_str_list(a: np.ndarray) -> List[str]:
    if a.ndim == 1:
        return [str(x).strip() for x in a.tolist()]
    if a.dtype.kind in {"U", "S"}:
        return ["".join(row).strip() for row in a]
    if a.dtype.kind in {"i", "u"}:
        out = []
        for row in a:
            chars = [chr(int(x)) for x in row if int(x) != 0]
            out.append("".join(chars).strip())
        return out
    return [str(x).strip() for x in a.astype(str)]


def load_uf_erdos992_from_mat(path: Path, one_indexed: bool = True) -> Tuple[nx.Graph, Dict[int, str]]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Problem" not in mat:
        raise ValueError("MAT file does not contain 'Problem'.")

    prob = mat["Problem"]
    A = prob.A
    G0 = nx.from_scipy_sparse_array(A, parallel_edges=False, create_using=nx.Graph)

    author_map = {}

    if hasattr(prob, "nodename"):
        raw = prob.nodename
        name_list = _mat_char_matrix_to_str_list(np.array(raw))
        for i, s in enumerate(name_list):
            nid = i + 1 if one_indexed else i
            author_map[nid] = s.strip()
    elif hasattr(prob, "aux") and hasattr(prob.aux, "nodename"):
        raw = prob.aux.nodename
        name_list = _mat_char_matrix_to_str_list(np.array(raw))
        for i, s in enumerate(name_list):
            nid = i + 1 if one_indexed else i
            author_map[nid] = s.strip()

    if one_indexed:
        mapping = {i: i + 1 for i in range(G0.number_of_nodes())}
        G = nx.relabel_nodes(G0, mapping, copy=True)
    else:
        G = G0

    return G, author_map


def simplify_graph(G: nx.Graph) -> nx.Graph:
    G = G.copy()
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    if G.is_directed():
        G = G.to_undirected(as_view=False)
    if isinstance(G, nx.MultiGraph):
        G = nx.Graph(G)
    return G


def giant_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G.copy()
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def communities_to_partition(communities: List[set]) -> Dict[int, int]:
    part = {}
    for cid, comm in enumerate(communities, start=1):
        for node in comm:
            part[node] = cid
    return part


def relabel_partition_by_size(partition: Dict[int, int]) -> Dict[int, int]:
    sizes = Counter(partition.values())
    ordered = sorted(sizes.items(), key=lambda x: (-x[1], x[0]))
    mapping = {old_cid: new_cid for new_cid, (old_cid, _) in enumerate(ordered, start=1)}
    return {node: mapping[cid] for node, cid in partition.items()}


def run_louvain_unbounded(G: nx.Graph, seed: int = DEFAULT_SEED) -> List[set]:
    return list(nx.community.louvain_communities(G, resolution=1.0, seed=seed))


def cross_edge_counter(G: nx.Graph, partition: Dict[int, int]) -> Counter:
    counts = Counter()
    for u, v in G.edges():
        cu, cv = partition[u], partition[v]
        if cu != cv:
            a, b = sorted((cu, cv))
            counts[(a, b)] += 1
    return counts


def merge_partition_to_k(G: nx.Graph, partition: Dict[int, int], target_k: int) -> Dict[int, int]:
    part = dict(partition)

    while len(set(part.values())) > target_k:
        counts = cross_edge_counter(G, part)
        if not counts:
            break

        (a, b), _ = max(counts.items(), key=lambda x: x[1])

        for node, cid in list(part.items()):
            if cid == b:
                part[node] = a

        current = sorted(set(part.values()))
        remap = {old: new for new, old in enumerate(current, start=1)}
        part = {node: remap[cid] for node, cid in part.items()}

    return relabel_partition_by_size(part)


def internal_degree(G: nx.Graph, node: int, partition: Dict[int, int]) -> int:
    cid = partition[node]
    return sum(1 for nbr in G.neighbors(node) if partition[nbr] == cid)


def community_topology_table(
    G: nx.Graph,
    partition: Dict[int, int],
    top_k_hubs: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, list]]:
    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    rows = []
    hub_text = {}
    hub_rows = {}

    for cid in sorted(communities):
        nodes = communities[cid]
        sub = G.subgraph(nodes)

        n_nodes = sub.number_of_nodes()
        internal_edges = sub.number_of_edges()

        total_deg_sum = sum(G.degree(n) for n in nodes)
        internal_pct = 0.0 if total_deg_sum == 0 else (2 * internal_edges) / total_deg_sum

        cut_edges = 0
        for n in nodes:
            for nbr in G.neighbors(n):
                if partition[nbr] != cid:
                    cut_edges += 1

        vol_s = total_deg_sum
        vol_not_s = 2 * G.number_of_edges() - vol_s
        denom = min(vol_s, vol_not_s)
        conductance = (cut_edges / denom) if denom > 0 else 0.0

        avg_clustering = nx.average_clustering(sub) if n_nodes > 1 else 0.0

        hub_candidates = []
        for n in nodes:
            ideg = internal_degree(G, n, partition)
            tdeg = G.degree(n)
            pct = 0.0 if tdeg == 0 else ideg / tdeg
            hub_candidates.append((n, ideg, tdeg, pct))

        hub_candidates.sort(key=lambda x: x[1], reverse=True)
        top_hubs = hub_candidates[:top_k_hubs]

        top_hub_nodes = [x[0] for x in top_hubs]
        possible_pairs = len(top_hub_nodes) * (len(top_hub_nodes) - 1) // 2
        actual_pairs = 0
        for u, v in combinations(top_hub_nodes, 2):
            if G.has_edge(u, v):
                actual_pairs += 1
        hub_interconn = f"{actual_pairs}/{possible_pairs}" if possible_pairs > 0 else "0/0"

        comm_name = f"C{cid}"

        rows.append({
            "Comm": comm_name,
            "Nodes": n_nodes,
            "Internal Edges": internal_edges,
            "Internal%": round(100 * internal_pct, 1),
            "Conductance": round(conductance, 3),
            "Avg Clustering": round(avg_clustering, 3),
            "Hub Interconn": hub_interconn,
        })

        desc = []
        row_pack = []
        for n, ideg, tdeg, pct in top_hubs:
            desc.append(f"Node {n} ({ideg}/{tdeg}, {round(100 * pct)}%)")
            row_pack.append({
                "Comm": comm_name,
                "Node": n,
                "Internal Degree": ideg,
                "Total Degree": tdeg,
                "Internal Ratio %": round(100 * pct, 1),
            })
        hub_text[comm_name] = ", ".join(desc)
        hub_rows[comm_name] = row_pack

    df = pd.DataFrame(rows).sort_values(["Nodes", "Internal Edges"], ascending=[False, False]).reset_index(drop=True)
    return df, hub_text, hub_rows


def cross_community_edges(G: nx.Graph, partition: Dict[int, int], top_n: int = 3) -> Dict[str, List[str]]:
    pair_counts = Counter()

    for u, v in G.edges():
        cu, cv = partition[u], partition[v]
        if cu != cv:
            a, b = sorted((cu, cv))
            pair_counts[(a, b)] += 1

    by_comm = defaultdict(list)
    for (a, b), cnt in pair_counts.items():
        by_comm[a].append((b, cnt))
        by_comm[b].append((a, cnt))

    summary = {}
    for cid in sorted(set(partition.values())):
        items = sorted(by_comm[cid], key=lambda x: x[1], reverse=True)[:top_n]
        summary[f"C{cid}"] = [f"C{cid} ↔ C{other}: {cnt:,}" for other, cnt in items]

    return summary


def classify_community_archetypes(df: pd.DataFrame, hub_text: Dict[str, str]) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        cid = row["Comm"]
        internal_pct = row["Internal%"]
        clustering = row["Avg Clustering"]
        conductance = row["Conductance"]

        if internal_pct >= 80 and clustering >= 0.45:
            label = "Dense Collaborative"
        elif internal_pct >= 80 and clustering < 0.30:
            label = "Star/Hub-Dominated"
        elif internal_pct <= 70 and conductance >= 0.30:
            label = "Bridge Community"
        else:
            label = "Mixed"

        rows.append({
            "Comm": cid,
            "Archetype": label,
            "Top Hubs": hub_text.get(cid, ""),
        })

    return pd.DataFrame(rows)


def build_structural_explanation(
    graph_name: str,
    df: pd.DataFrame,
    hub_text: Dict[str, str],
    cross_edges: Dict[str, List[str]],
    top_n: int = 3,
) -> str:
    lines = []
    lines.append(f"# {graph_name}: Community Analysis")
    lines.append("")
    lines.append("## Structural interpretation")
    lines.append("")

    top_df = df.head(top_n)

    for _, row in top_df.iterrows():
        cid = row["Comm"]
        nodes = int(row["Nodes"])
        internal_pct = row["Internal%"]
        conductance = row["Conductance"]
        clustering = row["Avg Clustering"]
        hub_interconn = row["Hub Interconn"]

        sent = (
            f"- **{cid}** has {nodes:,} nodes, internal% = {internal_pct:.1f}, "
            f"conductance = {conductance:.3f}, avg clustering = {clustering:.3f}, "
            f"and hub interconnection = {hub_interconn}. "
        )

        if internal_pct >= 80 and clustering >= 0.45:
            sent += (
                "This suggests a dense and cohesive community: nodes belong together "
                "because they share many internal ties and their local neighborhoods overlap strongly. "
            )
        elif internal_pct >= 80 and clustering < 0.30:
            sent += (
                "This suggests a hub-dominated or star-like community: many nodes belong together "
                "because they attach to a dominant internal hub rather than to each other. "
            )
        elif internal_pct <= 70 and conductance >= 0.30:
            sent += (
                "This suggests a bridge-like community: nodes are grouped together, but many edges also point outward, "
                "so the community is less self-contained and more connected to other regions of the graph. "
            )
        else:
            sent += (
                "This suggests a mixed mesoscopic structure with both internal cohesion and external connectivity. "
            )

        if cid in cross_edges and cross_edges[cid]:
            sent += f"Its strongest external links are: {', '.join(cross_edges[cid])}. "

        if cid in hub_text and hub_text[cid]:
            sent += f"Top internal hubs: {hub_text[cid]}."

        lines.append(sent)
        lines.append("")

    lines.append("## General answer to: Why do certain nodes belong to the same community?")
    lines.append("")
    lines.append(
        "- Nodes tend to belong to the same community when they share many internal edges, "
        "common neighbors, and short paths within the same local region."
    )
    lines.append(
        "- High internal% and high clustering usually indicate that nodes are grouped together "
        "because they form a dense local circle."
    )
    lines.append(
        "- Low internal% but high conductance often indicates a bridge community, where nodes are "
        "still related but also connect strongly to other communities."
    )
    lines.append(
        "- When one or two nodes dominate the internal degree distribution, the community can be "
        "hub-centered: members belong together because they are attached to the same central connector."
    )
    lines.append("")

    return "\n".join(lines)


ERDOS_RESEARCH_HINTS = {
    "Ernst G. Straus": "number theory, combinatorics, geometry",
    "Joel H. Spencer": "probabilistic combinatorics, graph theory",
    "Ralph J. Faudree": "graph theory, extremal combinatorics",
    "Andras Gyarfas": "graph theory, combinatorics",
    "András Gyárfás": "graph theory, combinatorics",
    "Endre Szemeredi": "extremal combinatorics, number theory",
    "Endre Szemerédi": "extremal combinatorics, number theory",
    "Jeno Lehel": "graph theory, combinatorics",
    "Jenő Lehel": "graph theory, combinatorics",
    "Richard H. Schelp": "graph theory, combinatorics",
    "Michael S. Jacobson": "graph theory, combinatorics",
    "Vera T. Sos": "combinatorics, number theory",
    "Vera T. Sós": "combinatorics, number theory",
    "Andras Hajnal": "set theory, combinatorics",
    "András Hajnal": "set theory, combinatorics",
}


def top_authors_in_community(
    G: nx.Graph,
    partition: Dict[int, int],
    cid: int,
    author_map: Dict[int, str],
    top_n: int = 5,
) -> List[Tuple[int, str, int, str]]:
    nodes = [n for n, c in partition.items() if c == cid]
    ranked = sorted(nodes, key=lambda n: G.degree(n), reverse=True)[:top_n]

    out = []
    for n in ranked:
        author = author_map.get(n, str(n))
        hint = ERDOS_RESEARCH_HINTS.get(author, "")
        out.append((n, author, G.degree(n), hint))
    return out


def build_erdos_author_report(
    G: nx.Graph,
    partition: Dict[int, int],
    author_map: Dict[int, str],
    largest_k: int = 3,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, str]:
    sizes = Counter(partition.values())
    largest_cids = [cid for cid, _ in sorted(sizes.items(), key=lambda x: (-x[1], x[0]))[:largest_k]]

    rows = []
    lines = []
    lines.append("# Erdos992: Top Authors in Largest Communities")
    lines.append("")

    for cid in largest_cids:
        lines.append(f"## C{cid}")
        lines.append("")

        top_authors = top_authors_in_community(G, partition, cid, author_map, top_n=top_n)
        for node_id, author, degree, hint in top_authors:
            rows.append({
                "Community": f"C{cid}",
                "Node ID": node_id,
                "Author": author,
                "Degree": degree,
                "Research Hint": hint,
            })
            if hint:
                lines.append(f"- {author} (node {node_id}, degree {degree}): {hint}")
            else:
                lines.append(f"- {author} (node {node_id}, degree {degree})")
        lines.append("")
        lines.append(
            "Interpretation: if the top authors in the same large community share similar research areas "
            "(for example graph theory, combinatorics, or number theory), that supports the idea that "
            "the detected community reflects a meaningful collaboration subfield rather than a random partition."
        )
        lines.append("")

    return pd.DataFrame(rows), "\n".join(lines)


def save_cross_edges_csv(cross_edges: Dict[str, List[str]], path: Path) -> None:
    rows = []
    for comm, items in cross_edges.items():
        for rank, txt in enumerate(items, start=1):
            rows.append({
                "Comm": comm,
                "Rank": rank,
                "Cross Community Edge": txt,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def save_hubs_csv(hub_rows: Dict[str, list], path: Path) -> None:
    rows = []
    for _, lst in hub_rows.items():
        rows.extend(lst)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_membership_csv(partition: Dict[int, int], path: Path) -> None:
    rows = [{"Node": node, "Community": cid} for node, cid in sorted(partition.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def analyze_one_graph(
    graph_name: str,
    G: nx.Graph,
    out_dir: Path,
    seed: int,
    author_map: Dict[int, str] = None,
    target_k: int = None,
) -> None:
    author_map = author_map or {}

    G = simplify_graph(G)
    G = giant_component(G)

    communities = run_louvain_unbounded(G, seed=seed)
    partition = communities_to_partition(communities)
    partition = relabel_partition_by_size(partition)

    if target_k is not None and target_k > 0 and target_k < len(set(partition.values())):
        partition = merge_partition_to_k(G, partition, target_k)

    df, hub_text, hub_rows = community_topology_table(G, partition, top_k_hubs=5)
    cross_edges = cross_community_edges(G, partition, top_n=3)
    archetypes_df = classify_community_archetypes(df, hub_text)
    report_md = build_structural_explanation(graph_name, df, hub_text, cross_edges, top_n=3)

    prefix = f"{graph_name}_k{target_k}" if target_k is not None else f"{graph_name}_unbounded"

    df.to_csv(out_dir / f"{prefix}_community_topology.csv", index=False)
    save_hubs_csv(hub_rows, out_dir / f"{prefix}_community_hubs.csv")
    save_cross_edges_csv(cross_edges, out_dir / f"{prefix}_cross_community_edges.csv")
    archetypes_df.to_csv(out_dir / f"{prefix}_community_archetypes.csv", index=False)
    save_membership_csv(partition, out_dir / f"{prefix}_community_membership.csv")

    with open(out_dir / f"{prefix}_community_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    if graph_name.lower().startswith("erdos") and author_map:
        author_df, author_md = build_erdos_author_report(G, partition, author_map, largest_k=3, top_n=5)
        author_df.to_csv(out_dir / f"{prefix}_largest_communities_top5_authors.csv", index=False)
        with open(out_dir / f"{prefix}_largest_communities_top5_authors.md", "w", encoding="utf-8") as f:
            f.write(author_md)

    print(f"[done] {graph_name} -> {prefix}")
    print(f"       nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}, communities={len(set(partition.values()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facebook", default=DEFAULT_FILES["facebook"])
    ap.add_argument("--enron", default=DEFAULT_FILES["enron"])
    ap.add_argument("--erdos", default=DEFAULT_FILES["erdos992"])
    ap.add_argument("--out-dir", default="community_analysis_results")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--graphs",
        nargs="+",
        default=["facebook", "enron", "erdos992"],
        choices=["facebook", "enron", "erdos992"],
        help="which graphs to analyze",
    )
    ap.add_argument(
        "--target-k",
        type=int,
        default=None,
        help="optional bounded number of communities, e.g. 5 / 15 / 30",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "facebook" in args.graphs:
        G_fb = load_snap_edgelist_gz(Path(args.facebook))
        analyze_one_graph(
            graph_name="facebook",
            G=G_fb,
            out_dir=out_dir,
            seed=args.seed,
            author_map={},
            target_k=args.target_k,
        )

    if "enron" in args.graphs:
        G_en = load_snap_edgelist_gz(Path(args.enron))
        analyze_one_graph(
            graph_name="enron",
            G=G_en,
            out_dir=out_dir,
            seed=args.seed,
            author_map={},
            target_k=args.target_k,
        )

    if "erdos992" in args.graphs:
        G_er, author_map = load_uf_erdos992_from_mat(Path(args.erdos), one_indexed=True)
        analyze_one_graph(
            graph_name="erdos992",
            G=G_er,
            out_dir=out_dir,
            seed=args.seed,
            author_map=author_map,
            target_k=args.target_k,
        )

    print("\nSaved outputs in:", out_dir)


if __name__ == "__main__":
    main()