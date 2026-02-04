from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return np.array(
            self.model.encode(texts, show_progress_bar=True)
        )
