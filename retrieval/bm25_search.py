from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, documents):
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        return sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
