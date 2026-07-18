from chunking.hierarchical import HierarchicalChunker

text = """
# Introduction

Retrieval-Augmented Generation (RAG) combines retrieval with language models.

RAG improves factual accuracy.

## Background

Dense retrieval converts text into embeddings.

BM25 is a sparse retrieval algorithm.

1 Methodology

We compare different chunking strategies.

We evaluate retrieval performance.

RESULTS

Hierarchical chunking preserves document structure.

It improves contextual retrieval.
"""

chunker = HierarchicalChunker(
    max_chunk_size=500,
)

chunks = chunker.split(text)

print("=" * 80)
print("Hierarchical Chunking")
print("=" * 80)

for chunk in chunks:

    print(f"Chunk ID : {chunk.id}")
    print(f"Section  : {chunk.section}")
    print(f"Level    : {chunk.level}")
    print(f"Length   : {chunk.length}")
    print("-" * 80)
    print(chunk.text)
    print("=" * 80)