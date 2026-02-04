from openai import OpenAI, OpenAIError

class OpenAIEmbedder:
    def __init__(self, api_key, model="text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts):
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]

        except OpenAIError as e:
            raise RuntimeError(f"OpenAI Embedding Error: {str(e)}")
