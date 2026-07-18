from pathlib import Path
import csv


class CSVWriter:

    @staticmethod
    def write(results, output_file="experiments/output/retrieval_results.csv"):
        """
        Save experiment results to a CSV file.
        """

        output_path = Path(output_file)

        # Create the directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:

            writer = csv.writer(csv_file)

            # Header
            writer.writerow([
                "Experiment",
                "Dataset",
                "Chunker",
                "Retriever",
                "TopK",
                "Queries",
                "Recall",
                "Precision",
                "HitRate",
                "MRR",
                "Latency(ms)"
            ])

            # Rows
            for result in results:

                writer.writerow([
                    result.experiment_name,
                    result.dataset,
                    result.chunker,
                    result.retriever,
                    result.top_k,
                    result.num_queries,
                    f"{result.recall:.4f}",
                    f"{result.precision:.4f}",
                    f"{result.hit_rate:.4f}",
                    f"{result.mrr:.4f}",
                    f"{result.latency_ms:.2f}",
                ])

        print(f"\nResults saved to: {output_path}")