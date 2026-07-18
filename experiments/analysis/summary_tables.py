from pathlib import Path

import pandas as pd

from .utils import load_results

# =============================================================================
# Output Directory
# =============================================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "tables"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Function
# =============================================================================

def save_table(df: pd.DataFrame, filename: str):
    """
    Save a DataFrame as CSV.
    """

    path = OUTPUT_DIR / filename

    df.to_csv(path, index=False)

    print(f"Saved: {path}")


# =============================================================================
# Chunker Summary
# =============================================================================

def chunker_summary(df: pd.DataFrame):

    summary = (
        df.groupby("Chunker")
        .agg(
            Recall=("Recall", "mean"),
            Precision=("Precision", "mean"),
            HitRate=("HitRate", "mean"),
            MRR=("MRR", "mean"),
            Latency=("Latency(ms)", "mean"),
        )
        .round(4)
        .sort_values("Recall", ascending=False)
        .reset_index()
    )

    save_table(summary, "chunker_summary.csv")

    return summary


# =============================================================================
# Retriever Summary
# =============================================================================

def retriever_summary(df: pd.DataFrame):

    summary = (
        df.groupby("Retriever")
        .agg(
            Recall=("Recall", "mean"),
            Precision=("Precision", "mean"),
            HitRate=("HitRate", "mean"),
            MRR=("MRR", "mean"),
            Latency=("Latency(ms)", "mean"),
        )
        .round(4)
        .sort_values("Recall", ascending=False)
        .reset_index()
    )

    save_table(summary, "retriever_summary.csv")

    return summary


# =============================================================================
# Dataset Summary
# =============================================================================

def dataset_summary(df: pd.DataFrame):

    summary = (
        df.groupby("Dataset")
        .agg(
            Recall=("Recall", "mean"),
            Precision=("Precision", "mean"),
            HitRate=("HitRate", "mean"),
            MRR=("MRR", "mean"),
            Latency=("Latency(ms)", "mean"),
        )
        .round(4)
        .sort_values("Recall", ascending=False)
        .reset_index()
    )

    save_table(summary, "dataset_summary.csv")

    return summary


# =============================================================================
# Best Configurations
# =============================================================================

def best_configurations(df: pd.DataFrame):

    ranking = (
        df.sort_values(
            ["Recall", "MRR", "Precision"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    save_table(ranking, "best_configurations.csv")

    return ranking


# =============================================================================
# Markdown Report
# =============================================================================

def markdown_report(
    chunker_df: pd.DataFrame,
    retriever_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
):

    report = OUTPUT_DIR / "analysis_summary.md"

    best = ranking_df.iloc[0]

    with open(report, "w") as f:

        f.write("# Experimental Analysis Summary\n\n")

        f.write("## Best Overall Configuration\n\n")

        f.write(
            f"- Chunker: **{best['Chunker']}**\n"
        )

        f.write(
            f"- Retriever: **{best['Retriever']}**\n"
        )

        f.write(
            f"- Dataset: **{best['Dataset']}**\n"
        )

        f.write(
            f"- Recall: **{best['Recall']:.4f}**\n"
        )

        f.write(
            f"- Precision: **{best['Precision']:.4f}**\n"
        )

        f.write(
            f"- HitRate: **{best['HitRate']:.4f}**\n"
        )

        f.write(
            f"- MRR: **{best['MRR']:.4f}**\n\n"
        )

        f.write("## Best Chunking Strategy\n\n")

        best_chunker = chunker_df.iloc[0]

        f.write(
            f"- {best_chunker['Chunker']} "
            f"(Recall={best_chunker['Recall']:.4f})\n\n"
        )

        f.write("## Best Retrieval Method\n\n")

        best_retriever = retriever_df.iloc[0]

        f.write(
            f"- {best_retriever['Retriever']} "
            f"(Recall={best_retriever['Recall']:.4f})\n\n"
        )

        f.write("## Dataset Ranking\n\n")

        for _, row in dataset_df.iterrows():

            f.write(
                f"- {row['Dataset']} "
                f"(Recall={row['Recall']:.4f})\n"
            )

    print(f"Saved: {report}")


# =============================================================================
# Generate Everything
# =============================================================================

def generate_summary_tables():

    df = load_results()

    chunker = chunker_summary(df)

    retriever = retriever_summary(df)

    dataset = dataset_summary(df)

    ranking = best_configurations(df)

    markdown_report(
        chunker,
        retriever,
        dataset,
        ranking,
    )


if __name__ == "__main__":
    generate_summary_tables()