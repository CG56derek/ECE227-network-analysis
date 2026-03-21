# Local-Level Analysis
 
Centrality analysis and top-10% overlap for Facebook, Enron, and Erdős networks.
 
## Files
 
### Centrality Analysis (Q1)
 
- `parallel_worker.py` — Parallel betweenness centrality using `multiprocessing`
- `degree analysis.ipynb` — Computes degree, eigenvector, closeness, and betweenness centrality (top 5 per network), and validates key nodes with clustering coefficient and average neighbor degree
 
### Top-10% Overlap (Q2)
 
- `q2_overlap_and_plot.py` — Exact overlap between top-10% degree and betweenness sets for all three networks, with plots
- `q2_enron_exact.py` — Standalone exact betweenness for Enron GCC (replaces earlier approximate results)
 
## Dependencies
 
All scripts import GCC graphs from `preprocess.py`:
 
```python
from preprocess import Gcc_fb, Gcc_enron, Gcc_erdos
```
 
Requires `networkx`, `matplotlib`, `numpy`.
 
## Running
 
```bash
python degree analysis.ipynb        # top-5 centrality per network & clustering/neighbor degree for key nodes
python q2_overlap_and_plot.py  # overlap metrics + plots (saves to results/)
```
