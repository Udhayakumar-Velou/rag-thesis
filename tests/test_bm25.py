from chunking.fixed import FixedChunker
from retrieval.bm25 import BM25Retriever

text = """
Retrieval-Augmented Generation improves language models by retrieving external knowledge.

Chunking is one of the most important preprocessing steps.

Dense retrieval converts text into vector embeddings.

FAISS performs efficient nearest-neighbor search.

Cats are common household pets.

Dogs are loyal companions.

Artificial intelligence is transforming many industries.
"""

chunker = FixedChunker(
    chunk_size=120,
    chunk_overlap=20
)

chunks = chunker.split(text)

retriever = BM25Retriever(chunks)

query = "What is dense retrieval?"

results = retriever.retrieve(
    query=query,
    top_k=3
)

print("=" * 60)
print(f"Query: {query}")
print("=" * 60)

for result in results:
    print(f"Chunk {result.id}")
    print(result.text)
    print("-" * 60)