from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "multi-qa-MiniLM-L6-cos-v1",
        window_size: int = 100,
        step_size: int = 50,
    ):
        self.model = SentenceTransformer(model_name)
        self.window_size = window_size
        self.step_size = step_size

    def generate_embeddings(self, documents: List[str]):
        windows = []
        embeddings = []

        for document in documents:
            tokens = document.split()

            for i in range(0, len(tokens), self.step_size):
                window_tokens = tokens[i : i + self.window_size]
                window_text = " ".join(window_tokens)
                embedding = self.model.encode(window_text)
                embeddings.append(embedding.tolist())
                windows.append(window_text)
                """ break """
            """ break """

        return embeddings, windows

    def generate_query_embedding(self, text):
        return self.model.encode(text)
