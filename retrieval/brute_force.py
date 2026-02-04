import numpy as np

class BruteForceSearch:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query_embedding, k=5):
        scores = np.dot(self.embeddings, query_embedding)
        return scores.argsort()[-k:][::-1]
