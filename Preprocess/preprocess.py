import networkx as nx
from scipy.io import loadmat
from pathlib import Path
G_fb = nx.read_edgelist(
    "data/facebook_combined.txt",
    comments="#",      
    nodetype=int,      
    create_using=nx.Graph()   
)

G_enron = nx.read_edgelist(
    "data/email-Enron.txt",
    comments="#",
    nodetype=int,
    create_using=nx.Graph()   
)

mat = loadmat("data/Erdos992.mat", squeeze_me=True, struct_as_record=False)

A = mat["Problem"].A

G_erdos = nx.from_scipy_sparse_array(A, create_using=nx.Graph())

for G in [G_fb, G_enron, G_erdos]:
    G.remove_edges_from(nx.selfloop_edges(G))

def show_info(name, G):
    print(f"\n[{name}]")
    print("nodes =", G.number_of_nodes())
    print("edges =", G.number_of_edges())
    print("components =", nx.number_connected_components(G))
    print("density =", round(nx.density(G), 6))

    gcc_nodes = max(nx.connected_components(G), key=len)
    Gcc = G.subgraph(gcc_nodes).copy()
    print("GCC nodes =", Gcc.number_of_nodes())
    print("GCC edges =", Gcc.number_of_edges())
    return Gcc

Gcc_fb = show_info("Facebook", G_fb)
Gcc_enron = show_info("Enron", G_enron)
Gcc_erdos = show_info("Erdos", G_erdos)

out_dir = Path("processed")
out_dir.mkdir(exist_ok=True)

nx.write_gexf(G_fb, out_dir / "facebook.gexf")
nx.write_gexf(G_enron, out_dir / "enron_full.gexf")
nx.write_gexf(G_erdos, out_dir / "erdos_full.gexf")

nx.write_gexf(Gcc_enron, out_dir / "enron_gcc.gexf")
nx.write_gexf(Gcc_erdos, out_dir / "erdos_gcc.gexf")

print( out_dir.resolve())