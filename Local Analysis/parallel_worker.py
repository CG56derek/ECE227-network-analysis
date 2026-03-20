import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def _compute_partial_betweenness(args):
    G, sources = args
    return nx.betweenness_centrality_subset(G, sources=sources, targets=list(G.nodes()), normalized=False, weight=None)

def parallel_betweenness(G):
    num_cores = cpu_count()
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    num_chunks = max(num_cores * 4, 100)
    chunk_size = max(1, n_nodes // num_chunks)
    
    node_chunks = [nodes[i:i + chunk_size] for i in range(0, n_nodes, chunk_size)]
    pool_args = [(G, chunk) for chunk in node_chunks]

    final_betweenness = {n: 0.0 for n in nodes}
    
    
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(_compute_partial_betweenness, pool_args), 
                            total=len(pool_args), 
                            desc="  current", 
                            unit="unit"))

    for partial_b in results:
        for n, v in partial_b.items():
            final_betweenness[n] += v
            
    return final_betweenness