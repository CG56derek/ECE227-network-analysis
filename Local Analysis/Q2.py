# q2_overlap_and_plot.py
# Q2: Top-10% overlap between degree centrality and betweenness centrality
# Exact version for all three GCC graphs

import math
import time
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

# Import GCC graphs from preprocess.py (located under Project227/)
try:
    from Project227.preprocess import Gcc_fb, Gcc_enron, Gcc_erdos
except ModuleNotFoundError:
    # Fallback if this script is moved into the same folder as preprocess.py
    from preprocess import Gcc_fb, Gcc_enron, Gcc_erdos


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


def compute_overlap_metrics(G, name):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    k_top = max(1, math.ceil(0.1 * n))

    print(f"\n=== {name} ===")
    print(f"GCC nodes = {n}, edges = {m}, top10% k = {k_top}")

    # Degree centrality
    t0 = time.time()
    deg_cent = nx.degree_centrality(G)
    t1 = time.time()
    print(f"Degree centrality done in {t1 - t0:.2f}s")

    # Exact betweenness centrality
    t0 = time.time()
    bet_cent = nx.betweenness_centrality(G, normalized=True)
    t1 = time.time()
    print(f"Betweenness centrality (exact) done in {t1 - t0:.2f}s")

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
    print(f"Overlap ratio = {overlap_ratio:.4f}")
    print(f"Jaccard = {jaccard:.4f}")
    print("Sample overlap nodes (up to 10):", sorted(list(inter))[:10])

    return {
        "network": name,
        "nodes_gcc": n,
        "edges_gcc": m,
        "top_k_10pct": k_top,
        "betweenness_mode": "exact",
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio,
        "jaccard": jaccard,
    }


def print_summary(results, order=("Facebook", "Enron", "Erdos")):
    print("\n=== Summary ===")
    results_sorted = sorted(results, key=lambda x: order.index(x["network"]))
    for r in results_sorted:
        print(
            f"{r['network']}: n={r['nodes_gcc']}, k={r['top_k_10pct']}, "
            f"overlap={r['overlap_count']}, ratio={r['overlap_ratio']:.4f}, "
            f"jaccard={r['jaccard']:.4f}, bet={r['betweenness_mode']}"
        )


def save_summary_csv(results, out_path):
    """
    Save a simple CSV without pandas dependency.
    """
    header = [
        "network",
        "nodes_gcc",
        "edges_gcc",
        "top_k_10pct",
        "betweenness_mode",
        "overlap_count",
        "overlap_ratio",
        "jaccard",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            row = [
                str(r["network"]),
                str(r["nodes_gcc"]),
                str(r["edges_gcc"]),
                str(r["top_k_10pct"]),
                str(r["betweenness_mode"]),
                str(r["overlap_count"]),
                f"{r['overlap_ratio']:.6f}",
                f"{r['jaccard']:.6f}",
            ]
            f.write(",".join(row) + "\n")
    print(f"Saved summary CSV to: {out_path}")


def plot_overlap_results(results, out_dir, show=True):
    order = ["Facebook", "Enron", "Erdos"]
    results_sorted = sorted(results, key=lambda x: order.index(x["network"]))

    networks = [r["network"] for r in results_sorted]
    overlap_ratio = [r["overlap_ratio"] for r in results_sorted]
    jaccard = [r["jaccard"] for r in results_sorted]

    # Plot 1: Overlap Ratio
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(networks, overlap_ratio)
    plt.ylim(0, 1.0)
    plt.ylabel("Overlap Ratio (|D∩B| / k)")
    plt.title("Top-10% Degree vs Betweenness Overlap Across Networks")
    for b, v in zip(bars, overlap_ratio):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    ratio_path = out_dir / "q2_overlap_ratio.png"
    plt.savefig(ratio_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {ratio_path}")
    if show:
        plt.show()
    else:
        plt.close()

    # Plot 2: Jaccard Similarity
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(networks, jaccard)
    plt.ylim(0, 1.0)
    plt.ylabel("Jaccard Similarity")
    plt.title("Jaccard Similarity of Top-10% Central Nodes")
    for b, v in zip(bars, jaccard):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    jaccard_path = out_dir / "q2_jaccard.png"
    plt.savefig(jaccard_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {jaccard_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    # Q2 computations on GCCs
    # exact betweenness centrality for all three networks
    res_fb = compute_overlap_metrics(Gcc_fb, "Facebook")
    res_enron = compute_overlap_metrics(Gcc_enron, "Enron")
    res_erdos = compute_overlap_metrics(Gcc_erdos, "Erdos")

    results = [res_fb, res_enron, res_erdos]

    print_summary(results)
    save_summary_csv(results, OUT_DIR / "q2_overlap_summary.csv")
    plot_overlap_results(results, OUT_DIR, show=True)


if __name__ == "__main__":
    main()