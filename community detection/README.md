# Simple README

This folder contains two Python scripts for graph preprocessing and community detection.

## Files

- `preprocess.py`  
  Loads the Facebook, Enron, and Erdos datasets, removes self-loops, prints basic graph statistics, and exports graphs to the `processed/` folder in GEXF format.

- `communit_analysis.py`  
  Loads processed GCC graphs, runs community detection with igraph, and saves the detected community labels back to GEXF files.

## Requirements

Install the required packages before running:

```bash
pip install networkx scipy python-igraph
```

## Data

Place the input files in a folder named `data/`:

- `facebook_combined.txt`
- `email-Enron.txt`
- `Erdos992.mat`

## How to run

First preprocess the graphs:

```bash
python preprocess.py
```

Then run community detection:

```bash
python communit_analysis.py
```

## Output

The scripts save output files in the `processed/` folder.
These files can be opened in tools such as Gephi for visualization and further analysis.

## Note

`communit_analysis.py` expects GCC graph files in `processed/`.
Make sure the required `.gexf` files exist before running it.
