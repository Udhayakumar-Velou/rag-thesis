# Evaluating Chunking and Retrieval Strategies for Lightweight RAG Systems

This repository contains the implementation for my Master's thesis at ESILV.

## Thesis Objective

Evaluate the impact of different chunking strategies and retrieval methods on Retrieval-Augmented Generation (RAG) systems.

## Project Structure

```
rag-thesis/
├── chunking/
├── embeddings/
├── retrieval/
├── vectorstore/
├── llm/
├── evaluation/
├── experiments/
├── data/
├── results/
└── main.py
```

## Planned Experiments

### Chunking
- Fixed-size
- Recursive
- Semantic
- Hierarchical
- Adaptive

### Retrieval
- BM25
- Dense Retrieval
- Hybrid Retrieval

### Embedding Model
- intfloat/e5-base-v2

### Vector Database
- FAISS

### Evaluation
- BEIR
- RAGAS

## Status

- ✅ Project structure
- ✅ Environment setup
- ✅ Dependencies installed
- 🔄 Implementation in progress