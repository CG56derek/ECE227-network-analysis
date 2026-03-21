import math
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Assumes the following already exist from your preprocessing:
#   BASE_DIR
#   Gcc_fb
#   Gcc_enron
#   Gcc_erdos
# =========================================================
# Import GCC graphs from preprocess.py (located under Project227/)
try:
    from Project227.preprocess import Gcc_fb, Gcc_enron, Gcc_erdos
except ModuleNotFoundError:
    # Fallback if this script is moved into the same folder as preprocess.py
    from preprocess import Gcc_fb, Gcc_enron, Gcc_erdos
# -----------------------------
# 0) Config
# -----------------------------
EMPIRICAL_GRAPHS = {
    "Facebook": Gcc_fb,  # placeholder, replace with Gcc_fb when ready
    "Enron": Gcc_enron,
    "Erdos": Gcc_erdos,
}

# 先补 Facebook 和 Erdos；要全跑就改成 ["Facebook", "Enron", "Erdos"]
NETWORKS_TO_RUN = ["Facebook", "Erdos"]

# baseline 每个模型生成多少次，取平均
N_BASELINE_RUNS = 3

# WS rewiring probability (和你们 methods 一致)
WS_REWIRE_P = 0.1

# 大图时对路径相关指标用近似，避免跑太久
APPROX_PATH_THRESHOLD = 10000   # n > 10000 用近似
APPROX_APL_SAMPLE_NODES = 400
APPROX_DIAM_SAMPLE_NODES = 200

# 输出目录
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "baseline_compare"
OUT_DIR.mkdir(exist_ok=True)

# 为了复现
GLOBAL_SEED = 42


# -----------------------------
# 1) Helpers
# -----------------------------
def get_gcc(G: nx.Graph) -> nx.Graph:
    """Return GCC copy."""
    if nx.is_connected(G):
        return G.copy()
    gcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(gcc_nodes).copy()


def nearest_even(x: float) -> int:
    """Nearest even integer >= 2."""
    k = int(round(x))
    if k < 2:
        return 2
    if k % 2 == 0:
        return k
    # choose nearer even integer
    lower = k - 1
    upper = k + 1
    if abs(lower - x) <= abs(upper - x):
        return max(2, lower)
    return upper


def degree_array(G: nx.Graph) -> np.ndarray:
    return np.array([d for _, d in G.degree()], dtype=float)


def degree_distribution(G: nx.Graph):
    """Return unique degrees and P(k)."""
    deg = degree_array(G).astype(int)
    vals, counts = np.unique(deg, return_counts=True)
    pk = counts / counts.sum()
    return vals, pk


def degree_ccdf(G: nx.Graph):
    """Return degrees k and CCDF P(X >= k)."""
    deg = np.sort(degree_array(G).astype(int))
    vals = np.unique(deg)
    ccdf = np.array([(deg >= k).mean() for k in vals], dtype=float)
    return vals, ccdf


def approx_average_shortest_path_length(G: nx.Graph, sample_size=400, seed=42) -> float:
    """
    Approximate APL by sampling source nodes and averaging shortest-path
    distances from each sampled source to all reachable nodes in the GCC.
    """
    rng = np.random.default_rng(seed)
    H = get_gcc(G)
    nodes = list(H.nodes())
    if len(nodes) <= sample_size:
        sample_nodes = nodes
    else:
        sample_nodes = rng.choice(nodes, size=sample_size, replace=False).tolist()

    total_dist = 0
    total_pairs = 0
    for s in sample_nodes:
        lengths = nx.single_source_shortest_path_length(H, s)
        total_dist += sum(lengths.values())  # includes distance 0 to itself
        total_pairs += (len(lengths) - 1)    # exclude self-pair

    return total_dist / total_pairs


def approx_diameter_lower_bound(G: nx.Graph, sample_size=200, seed=42) -> int:
    """
    Sample-based lower bound on diameter:
    max eccentricity over sampled source nodes in GCC.
    """
    rng = np.random.default_rng(seed)
    H = get_gcc(G)
    nodes = list(H.nodes())
    if len(nodes) <= sample_size:
        sample_nodes = nodes
    else:
        sample_nodes = rng.choice(nodes, size=sample_size, replace=False).tolist()

    diam_lb = 0
    for s in sample_nodes:
        lengths = nx.single_source_shortest_path_length(H, s)
        ecc = max(lengths.values())
        diam_lb = max(diam_lb, ecc)
    return int(diam_lb)


def compute_path_metrics(G: nx.Graph, name="", seed=42):
    """
    Exact for smaller graphs, approximate for large graphs.
    Returns: apl, diameter, apl_mode, diam_mode
    """
    H = get_gcc(G)
    n = H.number_of_nodes()

    if n > APPROX_PATH_THRESHOLD:
        apl = approx_average_shortest_path_length(
            H, sample_size=APPROX_APL_SAMPLE_NODES, seed=seed
        )
        diam = approx_diameter_lower_bound(
            H, sample_size=APPROX_DIAM_SAMPLE_NODES, seed=seed
        )
        return apl, diam, "approx", "lower_bound"
    else:
        apl = nx.average_shortest_path_length(H)
        diam = nx.diameter(H)
        return apl, diam, "exact", "exact"


def compute_metrics(G: nx.Graph, label="", seed=42) -> dict:
    """Compute paper-friendly global metrics on GCC."""
    H = get_gcc(G)
    deg = degree_array(H)

    apl, diam, apl_mode, diam_mode = compute_path_metrics(H, name=label, seed=seed)

    mean_deg = deg.mean()
    std_deg = deg.std(ddof=0)
    var_mean = (deg.var(ddof=0) / mean_deg) if mean_deg > 0 else np.nan

    row = {
        "Label": label,
        "Nodes": H.number_of_nodes(),
        "Edges": H.number_of_edges(),
        "AvgDegree": mean_deg,
        "DegreeStd": std_deg,
        "VarMean": var_mean,
        "MaxDegree": deg.max(),
        "AvgClustering": nx.average_clustering(H),
        "Transitivity": nx.transitivity(H),
        "AvgShortestPath": apl,
        "Diameter": diam,
        "APLMode": apl_mode,
        "DiameterMode": diam_mode,
    }
    return row


# -----------------------------
# 2) Matched baseline graph generators
# -----------------------------
def build_er(n: int, m: int, seed: int) -> nx.Graph:
    """ER matched by expected edge count."""
    p = 2 * m / (n * (n - 1))
    return nx.gnp_random_graph(n, p, seed=seed)


def build_ba(n: int, avg_degree: float, seed: int) -> nx.Graph:
    """BA matched by approximate average degree (~2m_attach)."""
    m_attach = max(1, int(round(avg_degree / 2)))
    m_attach = min(m_attach, n - 1)
    return nx.barabasi_albert_graph(n, m_attach, seed=seed)


def build_ws(n: int, avg_degree: float, seed: int, p_rewire: float = 0.1) -> nx.Graph:
    """WS matched by nearest even k and fixed rewiring probability."""
    k = nearest_even(avg_degree)
    # WS requires k < n and even
    if k >= n:
        k = n - 1 if (n - 1) % 2 == 0 else n - 2
    if k < 2:
        k = 2
    if k % 2 == 1:
        k -= 1
    return nx.watts_strogatz_graph(n, k, p_rewire, seed=seed)


def generate_baselines_for_empirical(G_emp: nx.Graph, seed_base=42, n_runs=3):
    """Return dict of model_name -> list of generated graphs."""
    H = get_gcc(G_emp)
    n = H.number_of_nodes()
    m = H.number_of_edges()
    avg_degree = 2 * m / n

    models = {"ER": [], "BA": [], "WS": []}
    for r in range(n_runs):
        seed = seed_base + r
        models["ER"].append(build_er(n, m, seed))
        models["BA"].append(build_ba(n, avg_degree, seed))
        models["WS"].append(build_ws(n, avg_degree, seed, p_rewire=WS_REWIRE_P))
    return models


# -----------------------------
# 3) Baseline comparison runner
# -----------------------------
def baseline_comparison_for_network(name: str, G_emp: nx.Graph):
    """
    Compute empirical metrics + averaged baseline metrics.
    Save:
      - raw per-run csv
      - averaged csv
      - paper-ready compact csv
      - figures
    """
    print(f"\n===== Running baseline comparison for {name} =====")

    # empirical
    empirical_row = compute_metrics(G_emp, label=name, seed=GLOBAL_SEED)

    # baselines
    model_graphs = generate_baselines_for_empirical(
        G_emp, seed_base=GLOBAL_SEED, n_runs=N_BASELINE_RUNS
    )

    all_rows = [empirical_row]
    baseline_run_rows = []

    for model_name, graphs in model_graphs.items():
        for i, Gm in enumerate(graphs, start=1):
            row = compute_metrics(Gm, label=f"{model_name}_run{i}", seed=GLOBAL_SEED + i)
            row["Model"] = model_name
            row["Run"] = i
            baseline_run_rows.append(row)

    baseline_runs_df = pd.DataFrame(baseline_run_rows)

    # average baseline metrics over runs
    numeric_cols = [
        "Nodes", "Edges", "AvgDegree", "DegreeStd", "VarMean", "MaxDegree",
        "AvgClustering", "Transitivity", "AvgShortestPath", "Diameter"
    ]
    baseline_avg_rows = []
    for model_name in ["ER", "BA", "WS"]:
        sub = baseline_runs_df[baseline_runs_df["Model"] == model_name]
        avg_row = {"Label": model_name}
        for c in numeric_cols:
            avg_row[c] = sub[c].mean()
        avg_row["APLMode"] = ",".join(sorted(set(sub["APLMode"])))
        avg_row["DiameterMode"] = ",".join(sorted(set(sub["DiameterMode"])))
        baseline_avg_rows.append(avg_row)

    comparison_df = pd.DataFrame([empirical_row] + baseline_avg_rows)

    # paper-friendly compact table
    paper_cols = [
        "Label", "AvgDegree", "DegreeStd", "MaxDegree",
        "AvgClustering", "AvgShortestPath", "Diameter"
    ]
    paper_df = comparison_df[paper_cols].copy()

    # round for paper
    for c in ["AvgDegree", "DegreeStd", "AvgClustering", "AvgShortestPath"]:
        paper_df[c] = paper_df[c].round(3)
    paper_df["MaxDegree"] = paper_df["MaxDegree"].round(0).astype(int)
    paper_df["Diameter"] = paper_df["Diameter"].round(2)

    # save tables
    baseline_runs_df.to_csv(OUT_DIR / f"{name.lower()}_baseline_runs_raw.csv", index=False)
    comparison_df.to_csv(OUT_DIR / f"{name.lower()}_baseline_compare_full.csv", index=False)
    paper_df.to_csv(OUT_DIR / f"{name.lower()}_baseline_compare_paper.csv", index=False)

    # save plots
    save_degree_compare_plot(name, G_emp, model_graphs)
    save_ccdf_compare_plot(name, G_emp, model_graphs)
    save_small_world_signature_plot(name, comparison_df)

    print(paper_df)
    print(f"Saved outputs to: {OUT_DIR.resolve()}")
    return comparison_df, paper_df


# -----------------------------
# 4) Plotting
# -----------------------------
def average_degree_distribution(graphs):
    """
    Average P(k) over repeated random graphs.
    Return unified k-grid and mean pk across runs.
    """
    all_deg = set()
    pk_maps = []
    for G in graphs:
        k, pk = degree_distribution(get_gcc(G))
        d = dict(zip(k, pk))
        pk_maps.append(d)
        all_deg.update(k.tolist())

    k_grid = np.array(sorted(all_deg))
    pk_mean = []
    for kk in k_grid:
        vals = [d.get(kk, 0.0) for d in pk_maps]
        pk_mean.append(np.mean(vals))
    return k_grid, np.array(pk_mean)


def average_ccdf(graphs):
    all_deg = set()
    ccdf_maps = []
    for G in graphs:
        k, cc = degree_ccdf(get_gcc(G))
        d = dict(zip(k, cc))
        ccdf_maps.append(d)
        all_deg.update(k.tolist())

    k_grid = np.array(sorted(all_deg))
    ccdf_mean = []
    for kk in k_grid:
        vals = [d.get(kk, 0.0) for d in ccdf_maps]
        ccdf_mean.append(np.mean(vals))
    return k_grid, np.array(ccdf_mean)


def save_degree_compare_plot(name, G_emp, model_graphs):
    plt.figure(figsize=(7, 5))

    # empirical
    k_emp, pk_emp = degree_distribution(get_gcc(G_emp))
    plt.scatter(k_emp, pk_emp, s=12, label=name)

    # baselines (mean over runs)
    for model_name in ["ER", "BA", "WS"]:
        k, pk = average_degree_distribution(model_graphs[model_name])
        plt.scatter(k, pk, s=12, label=model_name, alpha=0.85)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel("P(k)")
    plt.title(f"{name}: Degree Distribution (log-log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name.lower()}_degree_loglog_compare.png", dpi=300)
    plt.close()


def save_ccdf_compare_plot(name, G_emp, model_graphs):
    plt.figure(figsize=(7, 5))

    # empirical
    k_emp, cc_emp = degree_ccdf(get_gcc(G_emp))
    plt.plot(k_emp, cc_emp, linewidth=1.8, label=name)

    # baselines
    for model_name in ["ER", "BA", "WS"]:
        k, cc = average_ccdf(model_graphs[model_name])
        plt.plot(k, cc, linewidth=1.5, label=model_name, alpha=0.9)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel("P(X ≥ k)")
    plt.title(f"{name}: CCDF (log-log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name.lower()}_ccdf_compare.png", dpi=300)
    plt.close()


def save_small_world_signature_plot(name, comparison_df):
    """
    x = AvgShortestPath
    y = AvgClustering
    """
    plt.figure(figsize=(6.4, 5.0))

    color_map = {
        name: "black",
        "ER": "tomato",
        "BA": "dodgerblue",
        "WS": "mediumseagreen",
    }

    for _, row in comparison_df.iterrows():
        label = row["Label"]
        x = row["AvgShortestPath"]
        y = row["AvgClustering"]
        plt.scatter(x, y, s=260, color=color_map.get(label, "gray"),
                    edgecolors="black", linewidths=1.2, zorder=3)
        plt.text(x + 0.02, y + 0.01, label, fontsize=11, weight="bold")

    plt.xlabel("Average Shortest Path Length")
    plt.ylabel("Average Clustering Coefficient")
    plt.title(f"{name}: Small-World Signature")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name.lower()}_smallworld_signature.png", dpi=300)
    plt.close()


# -----------------------------
# 5) Run everything
# -----------------------------
all_paper_tables = []

for net_name in NETWORKS_TO_RUN:
    G_emp = EMPIRICAL_GRAPHS[net_name]
    full_df, paper_df = baseline_comparison_for_network(net_name, G_emp)

    # add network name for combined summary
    tmp = paper_df.copy()
    tmp.insert(0, "Network", net_name)
    all_paper_tables.append(tmp)

if all_paper_tables:
    combined_paper_df = pd.concat(all_paper_tables, ignore_index=True)
    combined_paper_df.to_csv(OUT_DIR / "combined_paper_summary.csv", index=False)
    print("\nCombined paper summary:")
    print(combined_paper_df)

print(f"\nDone. All outputs saved under:\n{OUT_DIR.resolve()}")