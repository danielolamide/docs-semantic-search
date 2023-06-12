from typing import List
import numpy as np
import chromadb
from chromadb.config import Settings


class DatabaseHandler:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

    def insert_documents(self, documents: List[str], embeddings):
        print("inserting embeddings")
        count = 0
        for doc, embedding in zip(documents, embeddings):
            self.collection.add(
                ids=[str(count)], embeddings=[embedding], documents=[doc]
            )
            count += 1

    def get_best_match(self, query_embedding: np.ndarray):
        embedding_list = query_embedding.tolist()
        best_match = self.collection.query(
            query_embeddings=embedding_list,
            n_results=1,
        )
        return best_match
