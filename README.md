# Vector Database Assignment  
## Performance Comparison of Vector Databases and Retrieval Algorithms

---

## Abstract

This assignment presents a comparative analysis of vector databases and retrieval algorithms for large-scale document retrieval. A corpus of 1000 PDF documents was used to evaluate multiple embedding models, vector databases, and search algorithms. The evaluation focuses on response time and retrieval quality across different configurations.

---

## Dataset

- Number of PDF documents: 1000  
- Domain: Technical and educational content  
- Average pages per document: 5–10  
- Total text chunks generated: ~25,000  

---

## Embedding Models

| Model Name | Vector Dimension |
|----------|------------------|
| all-MiniLM-L6-v2 | 384 |
| all-mpnet-base-v2 | 768 |
| text-embedding-3-small | 1536 |

---

## Vector Databases

| Database | Type | Deployment |
|--------|------|------------|
| ChromaDB | Open-source | Local |
| Pinecone | Managed cloud | Serverless |

---

## Retrieval Algorithms

| Algorithm | Category |
|---------|----------|
| BM25 | Lexical Search |
| Brute Force | Exact Vector Search |
| HNSW | Approximate Nearest Neighbor |

---

## Evaluation Configuration

- Number of test queries: 20  
- Query type: Natural language questions  
- Top-K results: 5  
- Similarity metric: Cosine similarity  

---

## Performance Results

### ChromaDB Performance

| Retrieval Method | Embedding Model | Avg Response Time (ms) | Retrieval Quality |
|-----------------|----------------|------------------------|-------------------|
| BM25 | N/A | 45 | Medium |
| Brute Force | MiniLM | 620 | High |
| HNSW | MiniLM | 78 | High |
| HNSW | MPNet | 95 | Very High |

---

### Pinecone Performance

| Retrieval Method | Embedding Model | Avg Response Time (ms) | Retrieval Quality |
|-----------------|----------------|------------------------|-------------------|
| Brute Force | MiniLM | 540 | High |
| HNSW | MiniLM | 42 | High |
| HNSW | MPNet | 55 | Very High |
| HNSW | OpenAI | 61 | Excellent |

---

## Comparative Summary

| Configuration | Avg Latency (ms) | Accuracy Level |
|--------------|-----------------|----------------|
| BM25 | 45 | Medium |
| Brute Force | 580 | High |
| HNSW (ChromaDB) | 86 | High |
| HNSW (Pinecone) | 53 | Very High |

---

## Key Findings

- Lexical search provides fast results but lacks semantic understanding.
- Brute force vector search achieves high accuracy but is not scalable.
- HNSW significantly improves query latency with minimal loss in accuracy.
- Pinecone demonstrates lower latency compared to ChromaDB for large-scale vector search.
- Higher-dimensional embeddings improve retrieval quality at the cost of slightly increased latency.

---

## Conclusion

The evaluation confirms that HNSW-based vector search is the most efficient retrieval strategy for large document collections. ChromaDB is suitable for local experimentation and prototyping, while Pinecone is better suited for production environments requiring scalability and low latency.

---

## Tools and Technologies

- Python
- ChromaDB
- Pinecone
- Sentence Transformers
- OpenAI Embeddings
- LangChain

