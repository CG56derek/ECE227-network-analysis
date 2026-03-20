# Link Prediction Project

This project compares baseline link prediction methods and GraphSAGE on three graphs: Facebook, Enron, and Erdos.

## Files

- `data_utils.py`: load and preprocess graph data
- `split.py`: build train/validation/test edge splits
- `features.py`: build node features
- `baselines.py`: run baseline methods
- `model.py`: GraphSAGE model
- `train.py`: train GraphSAGE
- `community_eval.py`: evaluate intra/inter-community performance
- `compare_results.py`: merge baseline and GraphSAGE results
- `plot_results.py`: make result figures

## Basic workflow

```bash
python split.py
python baselines.py
python train.py
python community_eval.py
python compare_results.py
python plot_results.py
```

## Output folders

- `artifacts/splits`
- `artifacts/baselines`
- `artifacts/models`
- `artifacts/results`
- `artifacts/community`
- `artifacts/comparison`
- `artifacts/figures`
