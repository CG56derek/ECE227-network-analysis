# Simple README

This folder contains two Python scripts for graph preprocessing and community analysis.

## Files

### preprocess.py
Loads the Facebook, Enron, and Erdos datasets, removes self-loops, extracts the giant connected component (GCC), prints basic graph statistics, and exports processed graphs for later analysis.

### community_analysis.py
Loads the processed graph files, runs community detection, compares different algorithms, analyzes community structure, and exports the final graph with node attributes to GEXF format for visualization.

## Requirements

Install the required packages before running:

```bash
pip install networkx numpy scipy pandas
```

## Data

Place the input files in a folder named `data/`:

```text
facebook_combined.txt
email-Enron.txt
Erdos992.mat
```

## How to Run

First preprocess the graphs:

```bash
python preprocess.py
```

Then run community analysis:

```bash
python community_analysis.py
```

## Output

The scripts save output files in folders such as `processed/` or `output/`.
These files can be opened in tools such as Gephi for visualization and further analysis.

Typical outputs include:

- processed GCC graph files
- community statistics printed in the terminal
- `.gexf` files with community labels and node attributes

## Note

- Make sure the required input files exist in the `data/` folder before running the scripts.
- `community_analysis.py` expects the processed graph files to be available before execution.
- If your actual script names are different, replace the filenames in the commands above.
