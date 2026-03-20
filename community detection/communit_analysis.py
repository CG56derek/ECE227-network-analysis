import networkx as nx
import igraph as ig
import time
from pathlib import Path

def detect_communities_fast(G_nx, graph_name, out_dir, bounded_k_list=[5, 15, 30]):
    print(f"\n[{graph_name} - Fast Community Detection with igraph]")

    ig_G = ig.Graph.from_networkx(G_nx)

    print("  Running Louvain (Unbounded)...")
    start_time = time.time()
    louvain_partition = ig_G.community_multilevel()
    louvain_time = time.time() - start_time

    print(f"  -> Louvain: Found {len(louvain_partition)} communities. Q = {louvain_partition.modularity:.4f}. Time: {louvain_time:.4f}s")

    louvain_dict = {ig_G.vs[node_idx]['_nx_name']: comm_id
                    for comm_id, cluster in enumerate(louvain_partition)
                    for node_idx in cluster}
    nx.set_node_attributes(G_nx, louvain_dict, 'louvain_comm')

    print("  Running Fast Greedy Modularity (Building Dendrogram)...")
    start_time = time.time()

    dendrogram = ig_G.community_fastgreedy()
    build_time = time.time() - start_time
    print(f"  -> Dendrogram tree built in ONLY {build_time:.4f}s!")

    for k in bounded_k_list:
        try:

            bounded_partition = dendrogram.as_clustering(n=k)
            print(f"    -> Bounded (k={k}): Q = {bounded_partition.modularity:.4f}")

            greedy_dict = {ig_G.vs[node_idx]['_nx_name']: comm_id
                           for comm_id, cluster in enumerate(bounded_partition)
                           for node_idx in cluster}
            nx.set_node_attributes(G_nx, greedy_dict, f'greedy_bounded_{k}_comm')
        except Exception as e:
            print(f"    -> Bounded (k={k}): Failed ({e})")

    file_path = out_dir / f"{graph_name}_fast_communities.gexf"
    nx.write_gexf(G_nx, file_path)
    print(f"  Saved all results to: {file_path}")

if __name__ == "__main__":
    out_dir = Path("processed")
    print("Loading GCC graphs...")

    Gcc_erdos = nx.read_gexf(out_dir / "erdos_gcc.gexf")
    Gcc_fb = nx.read_gexf(out_dir / "facebook.gexf")
    Gcc_enron = nx.read_gexf(out_dir / "enron_gcc.gexf")

    graphs_to_process = {
        "Erdos_GCC": Gcc_erdos,
        "Facebook_GCC": Gcc_fb,
        "Enron_GCC": Gcc_enron
    }

    for name, graph in graphs_to_process.items():
        detect_communities_fast(graph, name, out_dir, bounded_k_list=[5, 15, 30])