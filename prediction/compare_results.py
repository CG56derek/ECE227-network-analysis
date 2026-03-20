from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def load_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    name_map = {
        "common_neighbors": "Common Neighbors",
        "jaccard": "Jaccard",
        "adamic_adar": "Adamic-Adar",
        "preferential_attachment": "Preferential Attachment",
        "GraphSAGE": "GraphSAGE",
    }

    if "method" in df.columns:
        df["method"] = df["method"].replace(name_map)

    if "model" in df.columns:
        df["method"] = df["model"].replace(name_map)
        df = df.drop(columns=["model"])

    return df


def load_and_merge_results(
    baseline_path: Path | str = "artifacts/baselines/baseline_results.csv",
    graphsage_path: Path | str = "artifacts/results/graphsage_results.csv",
) -> pd.DataFrame:
    baseline_df = load_csv(baseline_path)
    graphsage_df = load_csv(graphsage_path)

    baseline_df = normalize_method_names(baseline_df)
    graphsage_df = normalize_method_names(graphsage_df)


    keep_cols = ["graph", "method", "val_auc", "val_ap", "test_auc", "test_ap"]

    baseline_df = baseline_df[keep_cols]
    graphsage_df = graphsage_df[keep_cols]

    merged = pd.concat([baseline_df, graphsage_df], ignore_index=True)

    method_order = [
        "Common Neighbors",
        "Jaccard",
        "Adamic-Adar",
        "Preferential Attachment",
        "GraphSAGE",
    ]

    graph_order = ["Facebook", "Enron", "Erdos"]

    merged["graph"] = pd.Categorical(merged["graph"], categories=graph_order, ordered=True)
    merged["method"] = pd.Categorical(merged["method"], categories=method_order, ordered=True)

    merged = merged.sort_values(["graph", "method"]).reset_index(drop=True)
    return merged


def print_full_table(df: pd.DataFrame) -> None:
    print(
        f"\n{'Graph':<12} {'Method':<28} "
        f"{'Val AUC':>9} {'Val AP':>9} {'Test AUC':>10} {'Test AP':>9}"
    )
    print("-" * 82)

    for _, row in df.iterrows():
        print(
            f"{row['graph']:<12} "
            f"{row['method']:<28} "
            f"{row['val_auc']:>9.4f} "
            f"{row['val_ap']:>9.4f} "
            f"{row['test_auc']:>10.4f} "
            f"{row['test_ap']:>9.4f}"
        )


def build_best_method_table(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("graph")["test_ap"].idxmax()
    best_df = df.loc[idx].copy().reset_index(drop=True)
    best_df = best_df.sort_values("graph").reset_index(drop=True)
    return best_df


def print_best_method_table(df: pd.DataFrame) -> None:
    print(f"\nBest method by Test AP")
    print(f"{'Graph':<12} {'Best Method':<28} {'Test AUC':>10} {'Test AP':>9}")
    print("-" * 63)

    for _, row in df.iterrows():
        print(
            f"{row['graph']:<12} "
            f"{row['method']:<28} "
            f"{row['test_auc']:>10.4f} "
            f"{row['test_ap']:>9.4f}"
        )


def save_tables(
    merged_df: pd.DataFrame,
    best_df: pd.DataFrame,
    out_dir: Path | str = "artifacts/comparison",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "all_methods_comparison.csv"
    best_path = out_dir / "best_method_by_graph.csv"

    merged_df.to_csv(merged_path, index=False)
    best_df.to_csv(best_path, index=False)

    print(f"\nSaved merged comparison to: {merged_path}")
    print(f"Saved best-method table to: {best_path}")


def main() -> None:
    merged_df = load_and_merge_results(
        baseline_path="artifacts/baselines/baseline_results.csv",
        graphsage_path="artifacts/results/graphsage_results.csv",
    )

    print_full_table(merged_df)

    best_df = build_best_method_table(merged_df)
    print_best_method_table(best_df)

    save_tables(merged_df, best_df, out_dir="artifacts/comparison")


if __name__ == "__main__":
    main()
