from pinecone import Pinecone, ServerlessSpec

class PineconeStore:
    def __init__(self, index_name, dimension, api_key):
        self.pc = Pinecone(api_key=api_key)

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.pc.Index(index_name)

    def add(self, embeddings, documents):
        vectors = [
            (str(i), emb.tolist(), {"text": documents[i]})
            for i, emb in enumerate(embeddings)
        ]
        self.index.upsert(vectors=vectors)

    def search(self, query_embedding, k=5):
        return self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )
