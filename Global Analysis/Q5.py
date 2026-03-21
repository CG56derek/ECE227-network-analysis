import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
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

DEGREE_DIR = OUT_DIR / "degree_distribution"
DEGREE_DIR.mkdir(exist_ok=True)


graphs = {
    "Facebook": Gcc_fb,
    "Enron": Gcc_enron,
    "Erdos": Gcc_erdos,
}


def get_degree_sequence(G):
    return np.array([d for _, d in G.degree()])


def summarize_degree_distribution(name, G):
    degrees = get_degree_sequence(G)
    summary = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "std_degree": float(np.std(degrees)),
    }

    print(f"\n[{name}]")
    for k, v in summary.items():
        print(f"{k} = {v}")

    return summary


def plot_degree_histogram(name, G, bins=50):
    degrees = get_degree_sequence(G)

    plt.figure(figsize=(7, 5))
    plt.hist(degrees, bins=bins, edgecolor="black")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"{name} Degree Distribution")
    plt.tight_layout()
    plt.savefig(DEGREE_DIR / f"{name.lower()}_degree_hist.png", dpi=300)
    plt.close()


def plot_degree_loglog(name, G):
    degrees = get_degree_sequence(G)
    unique_deg, counts = np.unique(degrees, return_counts=True)

    mask = unique_deg > 0
    unique_deg = unique_deg[mask]
    counts = counts[mask]

    probs = counts / counts.sum()

    plt.figure(figsize=(7, 5))
    plt.scatter(unique_deg, probs, s=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree (log)")
    plt.ylabel("P(k) (log)")
    plt.title(f"{name} Degree Distribution (log-log)")
    plt.tight_layout()
    plt.savefig(DEGREE_DIR / f"{name.lower()}_degree_loglog.png", dpi=300)
    plt.close()


def save_degree_rank_plot(name, G):
    degrees = np.sort(get_degree_sequence(G))[::-1]
    ranks = np.arange(1, len(degrees) + 1)

    plt.figure(figsize=(7, 5))
    plt.scatter(ranks, degrees, s=10)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank (log)")
    plt.ylabel("Degree (log)")
    plt.title(f"{name} Degree Rank Plot")
    plt.tight_layout()
    plt.savefig(DEGREE_DIR / f"{name.lower()}_degree_rank.png", dpi=300)
    plt.close()


def save_summary_txt(all_summaries):
    out_file = DEGREE_DIR / "degree_summary.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for name, summary in all_summaries.items():
            f.write(f"[{name}]\n")
            for k, v in summary.items():
                f.write(f"{k} = {v}\n")
            f.write("\n")


def main():
    all_summaries = {}

    for name, G in graphs.items():
        summary = summarize_degree_distribution(name, G)
        all_summaries[name] = summary

        plot_degree_histogram(name, G)
        plot_degree_loglog(name, G)
        save_degree_rank_plot(name, G)

    save_summary_txt(all_summaries)
    print(f"\nSaved all degree distribution results to: {DEGREE_DIR.resolve()}")


if __name__ == "__main__":
    main()