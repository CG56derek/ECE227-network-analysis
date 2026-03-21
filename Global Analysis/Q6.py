import time
from pathlib import Path

import networkx as nx

# Import GCC graphs from preprocess.py
try:
    from Project227.preprocess import Gcc_fb, Gcc_enron, Gcc_erdos
except ModuleNotFoundError:
    from preprocess import Gcc_fb, Gcc_enron, Gcc_erdos


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

PATH_DIR = OUT_DIR / "path_metrics"
PATH_DIR.mkdir(exist_ok=True)


graphs = {
    "Facebook": Gcc_fb,
    "Enron": Gcc_enron,
    "Erdos": Gcc_erdos,
}


def compute_path_metrics(name, G):
    print(f"\n[{name}]")
    print("nodes =", G.number_of_nodes())
    print("edges =", G.number_of_edges())

    start = time.time()
    avg_spl = nx.average_shortest_path_length(G)
    t1 = time.time()
    print(f"average_shortest_path_length = {avg_spl}")
    print(f"time for average shortest path = {t1 - start:.2f} s")

    diameter = nx.diameter(G)
    t2 = time.time()
    print(f"diameter = {diameter}")
    print(f"time for diameter = {t2 - t1:.2f} s")

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "average_shortest_path_length": avg_spl,
        "diameter": diameter,
        "time_avg_shortest_path_sec": round(t1 - start, 2),
        "time_diameter_sec": round(t2 - t1, 2),
    }


def save_results(all_results):
    out_file = PATH_DIR / "path_metrics.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for name, result in all_results.items():
            f.write(f"[{name}]\n")
            for k, v in result.items():
                f.write(f"{k} = {v}\n")
            f.write("\n")


def main():
    all_results = {}

    for name, G in graphs.items():
        result = compute_path_metrics(name, G)
        all_results[name] = result

    save_results(all_results)
    print(f"\nSaved path metrics to: {PATH_DIR.resolve()}")


if __name__ == "__main__":
    main()