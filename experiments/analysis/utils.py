from pathlib import Path

import pandas as pd


RESULTS_FILE = (
    Path(__file__).resolve()
    .parent.parent
    / "output"
    / "retrieval_results.csv"
)


def load_results() -> pd.DataFrame:
    """
    Load experiment results from the CSV file.
    """

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}"
        )

    df = pd.read_csv(RESULTS_FILE)

    print("=" * 60)
    print("Experiment Results Loaded")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    print(df.columns.tolist())
    print()

    print(df.head())

    return df


if __name__ == "__main__":
    load_results()