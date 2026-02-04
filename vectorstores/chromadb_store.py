import chromadb

class ChromaStore:
    def __init__(self, collection_name):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, embeddings, documents):
        ids = [str(i) for i in range(len(documents))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )

    def search(self, query_embedding, k=5):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
