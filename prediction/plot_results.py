# plot_results.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



BASELINE_COMPARE_PATH = Path("artifacts/comparison/all_methods_comparison.csv")
COMMUNITY_PATH = Path("artifacts/community/community_eval_results.csv")
HISTORY_DIR = Path("artifacts/training_history")
OUT_DIR = Path("artifacts/figures")


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_main_results() -> pd.DataFrame:
    if not BASELINE_COMPARE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BASELINE_COMPARE_PATH}")
    return pd.read_csv(BASELINE_COMPARE_PATH)


def load_community_results() -> pd.DataFrame:
    if not COMMUNITY_PATH.exists():
        raise FileNotFoundError(f"Missing file: {COMMUNITY_PATH}")
    return pd.read_csv(COMMUNITY_PATH)


def load_training_history(graph_name: str) -> pd.DataFrame:
    path = HISTORY_DIR / f"graphsage_{graph_name.lower()}_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)



def ordered_main_df(df: pd.DataFrame) -> pd.DataFrame:
    graph_order = ["Facebook", "Enron", "Erdos"]
    method_order = [
        "Common Neighbors",
        "Jaccard",
        "Adamic-Adar",
        "Preferential Attachment",
        "GraphSAGE",
    ]

    df = df.copy()
    df["graph"] = pd.Categorical(df["graph"], categories=graph_order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["graph", "method"]).reset_index(drop=True)
    return df


def ordered_community_df(df: pd.DataFrame) -> pd.DataFrame:
    graph_order = ["Facebook", "Enron", "Erdos"]
    case_order = ["intra", "inter"]
    method_order = ["Adamic-Adar", "Preferential Attachment", "GraphSAGE"]

    df = df.copy()
    df["graph"] = pd.Categorical(df["graph"], categories=graph_order, ordered=True)
    df["community_case"] = pd.Categorical(df["community_case"], categories=case_order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["graph", "community_case", "method"]).reset_index(drop=True)
    return df


def plot_overall_test_ap(df: pd.DataFrame) -> None:
    df = ordered_main_df(df)

    graphs = ["Facebook", "Enron", "Erdos"]
    methods = [
        "Common Neighbors",
        "Jaccard",
        "Adamic-Adar",
        "Preferential Attachment",
        "GraphSAGE",
    ]

    x = np.arange(len(graphs))
    width = 0.16

    plt.figure(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []
        for graph in graphs:
            row = df[(df["graph"] == graph) & (df["method"] == method)].iloc[0]
            values.append(row["test_ap"])

        offset = (i - 2) * width
        plt.bar(x + offset, values, width=width, label=method)

    plt.xticks(x, graphs)
    plt.ylabel("Test AP")
    plt.ylim(0.0, 1.05)
    plt.title("Overall Link Prediction Performance (Test AP)")
    plt.legend()
    plt.tight_layout()

    save_path = OUT_DIR / "overall_test_ap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_overall_test_auc(df: pd.DataFrame) -> None:
    df = ordered_main_df(df)

    graphs = ["Facebook", "Enron", "Erdos"]
    methods = [
        "Common Neighbors",
        "Jaccard",
        "Adamic-Adar",
        "Preferential Attachment",
        "GraphSAGE",
    ]

    x = np.arange(len(graphs))
    width = 0.16

    plt.figure(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []
        for graph in graphs:
            row = df[(df["graph"] == graph) & (df["method"] == method)].iloc[0]
            values.append(row["test_auc"])

        offset = (i - 2) * width
        plt.bar(x + offset, values, width=width, label=method)

    plt.xticks(x, graphs)
    plt.ylabel("Test AUC")
    plt.ylim(0.0, 1.05)
    plt.title("Overall Link Prediction Performance (Test AUC)")
    plt.legend()
    plt.tight_layout()

    save_path = OUT_DIR / "overall_test_auc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_community_test_ap(df: pd.DataFrame) -> None:
    df = ordered_community_df(df)

    graphs = ["Facebook", "Enron", "Erdos"]
    cases = ["intra", "inter"]
    methods = ["Adamic-Adar", "Preferential Attachment", "GraphSAGE"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, graph in zip(axes, graphs):
        sub = df[df["graph"] == graph]

        x = np.arange(len(cases))
        width = 0.22

        for i, method in enumerate(methods):
            values = []
            for case in cases:
                row = sub[(sub["community_case"] == case) & (sub["method"] == method)].iloc[0]
                values.append(row["test_ap"])

            offset = (i - 1) * width
            ax.bar(x + offset, values, width=width, label=method)

        ax.set_xticks(x)
        ax.set_xticklabels(["Intra", "Inter"])
        ax.set_title(graph)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Community Case")

    axes[0].set_ylabel("Test AP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Community-Aware Link Prediction Performance (Test AP)", y=1.16)
    plt.tight_layout()

    save_path = OUT_DIR / "community_test_ap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")



def plot_community_test_auc(df: pd.DataFrame) -> None:
    df = ordered_community_df(df)

    graphs = ["Facebook", "Enron", "Erdos"]
    cases = ["intra", "inter"]
    methods = ["Adamic-Adar", "Preferential Attachment", "GraphSAGE"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, graph in zip(axes, graphs):
        sub = df[df["graph"] == graph]

        x = np.arange(len(cases))
        width = 0.22

        for i, method in enumerate(methods):
            values = []
            for case in cases:
                row = sub[(sub["community_case"] == case) & (sub["method"] == method)].iloc[0]
                values.append(row["test_auc"])

            offset = (i - 1) * width
            ax.bar(x + offset, values, width=width, label=method)

        ax.set_xticks(x)
        ax.set_xticklabels(["Intra", "Inter"])
        ax.set_title(graph)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Community Case")

    axes[0].set_ylabel("Test AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Community-Aware Link Prediction Performance (Test AUC)", y=1.16)
    plt.tight_layout()

    save_path = OUT_DIR / "community_test_auc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")



def plot_graphsage_val_ap_curves() -> None:
    graphs = ["Facebook", "Enron", "Erdos"]

    plt.figure(figsize=(9, 5))

    for graph in graphs:
        hist = load_training_history(graph)
        plt.plot(hist["epoch"], hist["val_ap"], label=graph)

    plt.xlabel("Epoch")
    plt.ylabel("Validation AP")
    plt.ylim(0.0, 1.05)
    plt.title("GraphSAGE Validation AP Curves")
    plt.legend()
    plt.tight_layout()

    save_path = OUT_DIR / "graphsage_val_ap_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_graphsage_loss_curves() -> None:
    graphs = ["Facebook", "Enron", "Erdos"]

    plt.figure(figsize=(9, 5))

    for graph in graphs:
        hist = load_training_history(graph)
        plt.plot(hist["epoch"], hist["train_loss"], label=graph)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("GraphSAGE Training Loss Curves")
    plt.legend()
    plt.tight_layout()

    save_path = OUT_DIR / "graphsage_loss_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_quick_summary(main_df: pd.DataFrame, community_df: pd.DataFrame) -> None:
    print("\nBest method by graph (overall, Test AP):")
    idx = main_df.groupby("graph")["test_ap"].idxmax()
    best_df = main_df.loc[idx].sort_values("graph")

    for _, row in best_df.iterrows():
        print(
            f"{row['graph']:<10} "
            f"{row['method']:<24} "
            f"Test AP={row['test_ap']:.4f}, Test AUC={row['test_auc']:.4f}"
        )

    print("\nCommunity-aware summary (Test AP):")
    for graph in ["Facebook", "Enron", "Erdos"]:
        sub = community_df[community_df["graph"] == graph]
        print(f"\n{graph}:")
        for case in ["intra", "inter"]:
            sub_case = sub[sub["community_case"] == case]
            best_row = sub_case.loc[sub_case["test_ap"].idxmax()]
            print(
                f"  {case:<5} best = {best_row['method']:<24} "
                f"AP={best_row['test_ap']:.4f}, AUC={best_row['test_auc']:.4f}"
            )


def main() -> None:
    ensure_out_dir()

    main_df = load_main_results()
    community_df = load_community_results()

    main_df = ordered_main_df(main_df)
    community_df = ordered_community_df(community_df)

    print_quick_summary(main_df, community_df)

    plot_overall_test_ap(main_df)
    plot_overall_test_auc(main_df)
    plot_community_test_ap(community_df)
    plot_community_test_auc(community_df)
    plot_graphsage_val_ap_curves()
    plot_graphsage_loss_curves()

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()