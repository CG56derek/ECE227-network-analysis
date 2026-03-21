import networkx as nx
import numpy as np
from collections import Counter
from networkx.algorithms.community import louvain_communities, label_propagation_communities, modularity
import time


def load_enron_gcc(path):
    G = nx.read_edgelist(path, comments='#', nodetype=int)
    sl = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(sl)
    gcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(gcc_nodes).copy()


def algorithm_comparison(G):
    print("=== Algorithm Comparison ===")

    t0 = time.time()
    comms_louvain = louvain_communities(G, resolution=1.0, seed=42)
    t_louvain = time.time() - t0
    mod_louvain = modularity(G, comms_louvain)
    print(f"Louvain: {len(comms_louvain)} communities, modularity={mod_louvain:.4f}, {t_louvain:.1f}s")

    t0 = time.time()
    comms_lpa = list(label_propagation_communities(G))
    t_lpa = time.time() - t0
    mod_lpa = modularity(G, comms_lpa)
    print(f"LPA:     {len(comms_lpa)} communities, modularity={mod_lpa:.4f}, {t_lpa:.1f}s")

    return comms_louvain


def resolution_sensitivity(G):
    print("\n=== Resolution Sensitivity ===")
    for res in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        comms = louvain_communities(G, resolution=res, seed=42)
        mod = modularity(G, comms)
        comms_sorted = sorted(comms, key=len, reverse=True)
        top3 = [len(c) for c in comms_sorted[:3]]
        print(f"res={res}: {len(comms)} comms, mod={mod:.4f}, top3={top3}")


def community_topology(G, comms, deg_cent, bet_cent, eig_cent, n_core=9):
    print(f"\n=== Community Topology (Top {n_core}) ===")
    comms = sorted(comms, key=len, reverse=True)

    partition = {}
    for cid, comm in enumerate(comms):
        for node in comm:
            partition[node] = cid

    for i in range(n_core):
        comm = comms[i]
        sub = G.subgraph(comm)
        n = sub.number_of_nodes()
        internal = sub.number_of_edges()
        total_inc = sum(G.degree(node) for node in comm)
        external = total_inc - 2 * internal
        internal_ratio = (2 * internal) / total_inc if total_inc > 0 else 0
        cond = external / total_inc if total_inc > 0 else 0
        clust = nx.average_clustering(sub)

        top5 = sorted(sub.degree(), key=lambda x: x[1], reverse=True)[:5]
        hub_nodes = [nd for nd, _ in top5]
        hub_edges = sum(1 for a in range(len(hub_nodes)) for b in range(a+1, len(hub_nodes))
                        if sub.has_edge(hub_nodes[a], hub_nodes[b]))

        print(f"\nC{i+1}: {n} nodes, {internal} internal edges")
        print(f"  internal_ratio={internal_ratio:.3f}, conductance={cond:.3f}, clustering={clust:.3f}")
        print(f"  hub_interconnection={hub_edges}/10")
        print(f"  top 5 hubs:")
        for node, deg in top5:
            total_deg = G.degree(node)
            int_pct = deg / total_deg * 100 if total_deg > 0 else 0
            print(f"    Node {node:5d}: internal={deg}, total={total_deg}, "
                  f"internal%={int_pct:.0f}%, bc={bet_cent[node]:.5f}, ec={eig_cent[node]:.5f}")

    return partition


def cross_community_edges(G, comms, partition, n_core=9):
    print(f"\n=== Cross-Community Edges ===")
    comms = sorted(comms, key=len, reverse=True)

    for i in range(n_core):
        cross = Counter()
        for node in comms[i]:
            for nb in G.neighbors(node):
                nc = partition[nb]
                if nc != i:
                    cross[nc] += 1
        top3 = cross.most_common(3)
        links = ", ".join(f"C{c+1}: {cnt}" for c, cnt in top3)
        print(f"  C{i+1} -> {links}")


def export_gexf(G, partition, core_number, clustering, deg_cent, bet_cent, eig_cent,
                output_path, n_core=15):
    for node in G.nodes():
        cid = partition[node]
        G.nodes[node]['modularity_class'] = cid if cid < n_core else 999
        G.nodes[node]['community'] = f"C{cid+1}" if cid < n_core else "Other"
        G.nodes[node]['degree'] = G.degree(node)
        G.nodes[node]['degree_centrality'] = round(deg_cent[node], 8)
        G.nodes[node]['betweenness'] = round(bet_cent[node], 8)
        G.nodes[node]['eigenvector'] = round(eig_cent[node], 8)
        G.nodes[node]['kcore'] = core_number[node]
        G.nodes[node]['clustering'] = round(clustering.get(node, 0), 6)

    nx.write_gexf(G, output_path)
    print(f"\nGEXF exported: {G.number_of_nodes()} nodes -> {output_path}")


if __name__ == '__main__':
    DATA_PATH = 'data/email-Enron.txt'
    G = load_enron_gcc(DATA_PATH)
    print(f"GCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, k=500)
    try:
        eig_cent = nx.eigenvector_centrality(G, max_iter=500, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        eig_cent = nx.pagerank(G)

    comms = algorithm_comparison(G)
    resolution_sensitivity(G)
    partition = community_topology(G, comms, deg_cent, bet_cent, eig_cent)
    cross_community_edges(G, sorted(comms, key=len, reverse=True), partition)

    core_number = nx.core_number(G)
    clustering = nx.clustering(G)
    export_gexf(G, partition, core_number, clustering,
                deg_cent, bet_cent, eig_cent, 'output/enron_gcc.gexf')
