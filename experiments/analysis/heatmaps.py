from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKER_NAMES = {
    "FixedChunker": "Fixed",
    "RecursiveChunker": "Recursive",
    "SemanticChunker": "Semantic",
    "HierarchicalChunker": "Hierarchical",
    "AdaptiveChunker": "Adaptive",
}

RETRIEVER_NAMES = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid",
}


def plot_heatmap(df: pd.DataFrame, metric: str, filename: str):
    """
    Create a heatmap of Chunker × Retriever for the selected metric.
    """

    pivot = (
        df.pivot_table(
            index="Chunker",
            columns="Retriever",
            values=metric,
            aggfunc="mean",
        )
        .rename(index=CHUNKER_NAMES)
        .rename(columns=lambda x: RETRIEVER_NAMES.get(str(x).lower(), x))
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(
        f"{metric} Heatmap",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                f"{pivot.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if pivot.iloc[i, j] < 0.6 else "black",
                fontsize=9,
                fontweight="bold",
            )

    cbar = fig.colorbar(im)
    cbar.set_label(metric)

    plt.tight_layout()

    png = OUTPUT_DIR / f"{filename}.png"
    pdf = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(png, dpi=300)
    fig.savefig(pdf)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    plt.close(fig)