from evaluation.dataset import EvaluationDataset

dataset = EvaluationDataset("Sample Dataset")

dataset.add_query(
    query="What is dense retrieval?",
    relevant_chunk_ids=[2],
)

dataset.add_query(
    query="What is BM25?",
    relevant_chunk_ids=[5],
)

dataset.add_query(
    query="What is hybrid retrieval?",
    relevant_chunk_ids=[8],
)

print("=" * 60)
print(dataset.name)
print("=" * 60)

for sample in dataset:
    print(sample.query)
    print(sample.relevant_chunk_ids)
    print("-" * 60)

print(f"Total Queries: {len(dataset)}")