from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)

retrieved = [4, 2, 8, 1, 7]
relevant = [2, 7]

print("=" * 50)

print("Recall@1")
print(recall_at_k(retrieved, relevant, 1))

print("Recall@3")
print(recall_at_k(retrieved, relevant, 3))

print("Recall@5")
print(recall_at_k(retrieved, relevant, 5))

print("=" * 50)

print("Precision@3")
print(precision_at_k(retrieved, relevant, 3))

print("Precision@5")
print(precision_at_k(retrieved, relevant, 5))

print("=" * 50)

print("Hit Rate@3")
print(hit_rate_at_k(retrieved, relevant, 3))

print("Hit Rate@5")
print(hit_rate_at_k(retrieved, relevant, 5))

print("=" * 50)

print("MRR")
print(mean_reciprocal_rank(retrieved, relevant))