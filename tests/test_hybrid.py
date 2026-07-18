from chunking.fixed import FixedChunker
from embeddings.e5_embedding import E5Embedding
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from vectorstore.faiss_store import FAISSStore

text = """
Retrieval-Augmented Generation improves language models by retrieving external knowledge.

Chunking is one of the most important preprocessing steps.

Dense retrieval converts text into vector embeddings.

BM25 is a sparse keyword retrieval algorithm.

Hybrid retrieval combines dense retrieval and BM25.

FAISS performs efficient similarity search.

Semantic chunking groups related sentences together.
"""

chunker = FixedChunker(
    chunk_size=120,
    chunk_overlap=20,
)

chunks = chunker.split(text)

embedding_model = E5Embedding()

vector_store = FAISSStore(embedding_model)
vector_store.add_documents(chunks)

dense = DenseRetriever(vector_store)
bm25 = BM25Retriever(chunks)

hybrid = HybridRetriever(
    dense_retriever=dense,
    bm25_retriever=bm25,
)

query = "What is hybrid retrieval?"

print("=" * 60)
print("Dense Retrieval")
print("=" * 60)

for chunk in dense.retrieve(query):
    print(f"Chunk ID : {chunk.id}")
    print(f"Length   : {chunk.length}")
    print(chunk.text)
    print("-" * 60)

print()

print("=" * 60)
print("BM25 Retrieval")
print("=" * 60)

for chunk in bm25.retrieve(query):
    print(f"Chunk ID : {chunk.id}")
    print(f"Length   : {chunk.length}")
    print(chunk.text)
    print("-" * 60)

print()

print("=" * 60)
print("Hybrid Retrieval (RRF)")
print("=" * 60)

for chunk in hybrid.retrieve(query):
    print(f"Chunk ID : {chunk.id}")
    print(f"Length   : {chunk.length}")
    print(chunk.text)
    print("-" * 60)