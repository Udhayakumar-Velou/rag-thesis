from chunking.adaptive import AdaptiveChunker

text = """
Retrieval-Augmented Generation improves language models.

Chunking is one of the most important preprocessing steps.

Dense retrieval converts text into vector embeddings.

BM25 is a sparse keyword retrieval algorithm.

Hybrid retrieval combines dense retrieval and BM25.

FAISS performs efficient similarity search.

Semantic chunking groups related sentences together.
"""

chunker = AdaptiveChunker(
    max_chunk_size=120,
    min_chunk_size=60,
)

chunks = chunker.split(text)

print("=" * 80)
print("Adaptive Chunking")
print("=" * 80)

for chunk in chunks:
    print(f"Chunk ID : {chunk.id}")
    print(f"Length   : {chunk.length}")
    print("-" * 80)
    print(chunk.text)
    print("=" * 80)