from embeddings.e5_embedding import E5Embedding

embedding_model = E5Embedding()

query = "What is Retrieval-Augmented Generation?"

document = """
Retrieval-Augmented Generation combines retrieval systems with
large language models to improve factual accuracy.
"""

query_embedding = embedding_model.embed_query(query)
document_embedding = embedding_model.embed_document(document)

print("Query shape:", query_embedding.shape)
print("Document shape:", document_embedding.shape)