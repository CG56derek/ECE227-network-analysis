# ECE 227 — Network Analysis and Visualization

Multi-scale structural analysis of three real-world networks (Facebook, Enron, Erdős) using NetworkX and Gephi, plus link prediction experiments comparing local heuristics with GraphSAGE.

> **Course:** ECE 227, UC San Diego  
> **Authors:** Ke Liu · Yi Huang · Chutian Gong

---

## Overview

We study three empirical networks at three levels of granularity:

| Network | Nodes | Edges | Source |
|---------|-------|-------|--------|
| Facebook ego | 4 039 | 88 234 | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) |
| Enron email | 36 692 | 183 831 | [SNAP](https://snap.stanford.edu/data/email-Enron.html) |
| Erdős collaboration | 6 100 | 7 515 | [SuiteSparse (Erdos992)](https://sparse.tamu.edu/) |

All graphs are converted to undirected simple graphs; analysis is performed on the giant connected component (GCC).

**Local level** — centrality measures (degree, eigenvector, closeness, betweenness) and top-10% degree–betweenness overlap.  
**Mesoscopic level** — community detection via Louvain and greedy modularity (k = 5, 15, 30), visualized with ForceAtlas2 in Gephi.  
**Global level** — degree distributions, average shortest path length, diameter, and comparison with ER / BA / WS random graph baselines.  
**Prediction** — link prediction using Common Neighbors, Jaccard, Adamic–Adar, Preferential Attachment, and GraphSAGE, with community-aware evaluation.

---

## Repository Structure

```
.
├── Preprocess/
│   └── preprocess.py              # Load datasets, remove self-loops, extract GCCs
│
├── Local Analysis/
│   ├── degree analysis.ipynb      # Centrality computation (top-5 per network)
│   ├── Q2.py                      # Top-10% degree vs betweenness overlap
│   ├── Q2-Enron.py                # Standalone exact betweenness for Enron GCC
│   └── parallel_worker.py         # Parallel betweenness via multiprocessing
│
├── Mesoscopic Analysis/
│   ├── community_analysis.py      # Louvain & greedy modularity partitions
│   ├── enron_community.py         # Enron-specific community analysis
│   └── data/                      # Raw datasets (Facebook, Enron, Erdős)
│
├── Global Analysis/
│   ├── Q5.py                      # Degree distribution stats & plots
│   ├── Q6.py                      # Average shortest path length & diameter
│   └── Q7.py                      # Comparison with ER / BA / WS baselines
│
├── community detection/
│   ├── preprocess.py              # Export GCCs to GEXF for Gephi
│   ├── communit_analysis.py       # igraph-based community detection → GEXF
│   └── data/                      # Raw datasets
│
├── prediction/
│   ├── data_utils.py              # Graph loading utilities
│   ├── split.py                   # Train/val/test edge split (8/1/1)
│   ├── features.py                # Structural node features
│   ├── baselines.py               # CN, Jaccard, Adamic–Adar, Pref. Attachment
│   ├── model.py                   # GraphSAGE (2-layer, dot-product decoder)
│   ├── train.py                   # GraphSAGE training loop
│   ├── community_eval.py          # Intra- vs inter-community evaluation
│   ├── compare_results.py         # Merge baseline & GraphSAGE results
│   ├── plot_results.py            # Generate result figures
│   └── data/                      # Raw datasets
│
└── README.md
```

---

## Requirements

- Python 3.9+
- Core: `networkx`, `numpy`, `scipy`, `matplotlib`
- Community detection: `python-igraph`
- Prediction: `torch`, `torch_geometric`
- Visualization: [Gephi](https://gephi.org/) (for ForceAtlas2 layout and GEXF rendering)

Install Python dependencies:

```bash
pip install networkx numpy scipy matplotlib python-igraph torch torch_geometric
```

---

## Quickstart

### 1. Preprocessing

```bash
cd Preprocess
python preprocess.py
```

This loads the three raw datasets, removes self-loops, and extracts the GCC for each network. Other scripts import the GCC objects directly from `preprocess.py`.

### 2. Local Analysis

```bash
cd "Local Analysis"
jupyter notebook "degree analysis.ipynb"   # centrality computation
python Q2.py                                # top-10% overlap
```

### 3. Mesoscopic Analysis

```bash
cd "Mesoscopic Analysis"
python community_analysis.py
```

For Gephi visualization, use the `community detection/` folder to export GEXF files:

```bash
cd "community detection"
python preprocess.py            # export GCCs as GEXF
python communit_analysis.py     # attach community labels
```

Then open the resulting `.gexf` files in Gephi and apply ForceAtlas2 layout.

### 4. Global Analysis

```bash
cd "Global Analysis"
python Q5.py    # degree distributions
python Q6.py    # shortest paths & diameter
python Q7.py    # random graph model comparison
```

### 5. Link Prediction

```bash
cd prediction
python split.py
python baselines.py
python train.py
python community_eval.py
python compare_results.py
python plot_results.py
```

Results and figures are saved under `prediction/artifacts/`.

---

## Key Findings

- **Local:** Degree/betweenness centrality tend to highlight the same "broker" nodes, while eigenvector centrality picks out nodes embedded in dense, high-prestige cores. The degree–betweenness overlap decreases from Erdős (0.85) → Enron (0.62) → Facebook (0.26), reflecting increasing community insularity.
- **Mesoscopic:** Facebook has the sharpest community boundaries (high internal ratio, low conductance). Enron communities overlap more due to cross-departmental communication. Erdős communities are hub-centered with low clustering.
- **Global:** All three networks show heavy-tailed degree distributions and small-world-like short paths, more consistent with BA-style heterogeneity than ER random graphs.
- **Prediction:** Adamic–Adar dominates on Facebook and Enron; Erdős favors Preferential Attachment and GraphSAGE, consistent with its hub-driven, low-clustering structure.

---

## Datasets

| File | Description |
|------|-------------|
| `facebook_combined.txt` | Facebook ego-network edge list |
| `email-Enron.txt` | Enron email communication edge list |
| `Erdos992.mat` | Erdős collaboration adjacency matrix (MATLAB sparse format) |

Each analysis folder contains its own copy of the raw data under `data/`.

---

## License

This project was developed for ECE 227 at UC San Diego. The datasets are publicly available from their original sources (see table above).
