# q2_enron_exact.py
# Exact top-10% overlap between degree centrality and betweenness centrality
# for the Enron GCC only

import math
import time
from pathlib import Path

import networkx as nx

# Import Enron GCC from preprocess.py
try:
    from Project227.preprocess import Gcc_enron
except ModuleNotFoundError:
    from preprocess import Gcc_enron


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


def top_k_nodes_from_dict(score_dict, k):
    """
    Deterministic top-k selection:
    sort by score descending, then node id ascending (tie-breaker).
    """
    ranked = sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))
    return [node for node, _ in ranked[:k]]


def compute_enron_overlap_exact(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k_top = max(1, math.ceil(0.1 * n))

    print("\n=== Enron (Exact) ===")
    print(f"GCC nodes = {n}, edges = {m}, top10% k = {k_top}")

    # Degree centrality
    t0 = time.time()
    print("Starting degree centrality...")
    deg_cent = nx.degree_centrality(G)
    t1 = time.time()
    print(f"Degree centrality done in {t1 - t0:.2f}s")

    # Exact betweenness centrality
    print("Starting betweenness centrality (exact)...")
    t2 = time.time()
    bet_cent = nx.betweenness_centrality(G, normalized=True)
    t3 = time.time()
    print(f"Betweenness centrality (exact) done in {t3 - t2:.2f}s")

    # Top 10% sets
    top_deg = set(top_k_nodes_from_dict(deg_cent, k_top))
    top_bet = set(top_k_nodes_from_dict(bet_cent, k_top))

    # Overlap metrics
    inter = top_deg & top_bet
    union = top_deg | top_bet

    overlap_count = len(inter)
    overlap_ratio = overlap_count / k_top
    jaccard = overlap_count / len(union) if len(union) > 0 else 0.0

    print(f"Overlap count = {overlap_count}")
    print(f"Overlap ratio = {overlap_ratio:.6f}")
    print(f"Jaccard = {jaccard:.6f}")
    print("Sample overlap nodes (up to 20):", sorted(list(inter))[:20])

    return {
        "network": "Enron",
        "nodes_gcc": n,
        "edges_gcc": m,
        "top_k_10pct": k_top,
        "betweenness_mode": "exact",
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio,
        "jaccard": jaccard,
        "degree_time_sec": round(t1 - t0, 2),
        "betweenness_time_sec": round(t3 - t2, 2),
        "total_time_sec": round(t3 - t0, 2),
    }


def save_result(result, out_dir):
    txt_path = out_dir / "q2_enron_exact_result.txt"
    csv_path = out_dir / "q2_enron_exact_result.csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[Enron]\n")
        for k, v in result.items():
            f.write(f"{k} = {v}\n")

    with open(csv_path, "w", encoding="utf-8") as f:
        header = list(result.keys())
        f.write(",".join(header) + "\n")
        row = [str(result[h]) for h in header]
        f.write(",".join(row) + "\n")

    print(f"Saved TXT to: {txt_path}")
    print(f"Saved CSV to: {csv_path}")


def main():
    result = compute_enron_overlap_exact(Gcc_enron)
    save_result(result, OUT_DIR)


if __name__ == "__main__":
    main()