# Evaluating Chunking and Retrieval Strategies for Lightweight Retrieval-Augmented Generation Systems

This repository contains the implementation developed for my Master's thesis at **ESILV – École Supérieure d'Ingénieurs Léonard de Vinci**.

The project presents a lightweight Retrieval-Augmented Generation (RAG) framework designed to systematically evaluate the impact of different document chunking strategies and retrieval methods on retrieval effectiveness and response quality.

---

# Thesis Objective

The objective of this research is to investigate how different document chunking strategies interact with various retrieval methods within a lightweight RAG pipeline.

The framework evaluates multiple combinations of chunking strategies and retrieval methods while keeping all other components constant, enabling a fair comparison of their impact on retrieval performance and answer generation quality.

---

# Framework Overview

The implemented RAG pipeline consists of the following stages:

```
Documents
    │
    ▼
Document Preprocessing
    │
    ▼
Document Chunking
    │
    ▼
E5-base-v2 Embedding
    │
    ▼
FAISS Vector Database
    │
──────────────────────────────────────────────
    │
    ▼
User Query
    │
    ▼
E5-base-v2 Embedding
    │
    ▼
Retriever
(BM25 | Dense | Hybrid)
    │
    ▼
Top-k Retrieved Chunks
    │
    ▼
Prompt Construction
    │
    ▼
Qwen2.5:7B (Ollama)
    │
    ▼
Generated Response
```

---

# Project Structure

```
rag-thesis/
│
├── chunking/
│   ├── fixed.py
│   ├── recursive.py
│   ├── semantic.py
│   ├── hierarchical.py
│   └── adaptive.py
│
├── embeddings/
│   └── e5_embedding.py
│
├── retrieval/
│   ├── bm25.py
│   ├── dense.py
│   └── hybrid.py
│
├── vectorstore/
│   └── faiss_store.py
│
├── llm/
│   └── ollama_llm.py
│
├── evaluation/
│   ├── retrieval_metrics.py
│   └── ragas_metrics.py
│
├── experiments/
│   ├── configs/
│   ├── experiment_runner.py
│   └── experiment_result.py
│
├── data/
│   └── datasets/
│
├── results/
│   └── logs/
│
├── main.py
└── README.md
```

---

# Experimental Configuration

## Document Chunking Strategies

- Fixed-size Chunking
- Recursive Chunking
- Semantic Chunking
- Hierarchical Chunking
- Adaptive Chunking

---

## Retrieval Methods

- BM25
- Dense Retrieval
- Hybrid Retrieval

---

## Embedding Model

- **intfloat/e5-base-v2**

---

## Vector Database

- **FAISS**

---

## Large Language Model

- **Qwen2.5:7B**
- **Ollama**

---

## Benchmark Datasets

The experiments are conducted using three datasets from the **BEIR benchmark**:

- Natural Questions (NQ)
- HotpotQA
- SciFact

---

## Evaluation Metrics

### Retrieval Metrics

- Recall@k
- Precision@k
- Hit Rate
- Mean Reciprocal Rank (MRR)
- Retrieval Latency

### Generation Metrics (RAGAS)

- Faithfulness
- Answer Relevancy

---

# Running the Experiments

Run all experiments using:

```bash
python main.py
```

The framework automatically evaluates every combination of:

- 5 Chunking Strategies
- 3 Retrieval Methods
- 3 BEIR Datasets

Total Experiments:

```
5 × 3 × 3 = 45 Experiments
```

---

# Results

Experimental results are automatically stored in:

```
results/logs/
```

Generated files:

```
experiment_results.csv
experiment_failures.csv
```

---

# Technologies Used

- Python
- LangChain
- FAISS
- Ollama
- Qwen2.5:7B
- Hugging Face Transformers
- E5-base-v2
- BEIR
- RAGAS

---

# Thesis Information

**Title**

> Evaluating Chunking and Retrieval Strategies for Lightweight Retrieval-Augmented Generation Systems

**Institution**

ESILV – École Supérieure d'Ingénieurs Léonard de Vinci

Master of Science in Data Science & Artificial Intelligence

---

# License

This repository is intended for academic research purposes.