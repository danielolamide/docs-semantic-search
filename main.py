from typing import List
from DataIngester import DataIngester
from EmbeddingGenerator import EmbeddingGenerator
from DatabaseHandler import DatabaseHandler


def main(filepaths: List[str]):
    window_size = 100
    step_size = 50
    documents = DataIngester(filepaths).ingest()
    embedding_generator = EmbeddingGenerator(
        window_size=window_size, step_size=step_size
    )
    db_handler = DatabaseHandler()

    print("ingesting and processing documents")
    embeddings, windows = embedding_generator.generate_embeddings(documents)
    db_handler.insert_documents(windows, embeddings)

    # User Interaction
    while True:
        query = input("Ask a question or type 'exit' to quit: ")

        if query.lower() == "exit":
            break

        # Generate query embedding
        query_embedding = embedding_generator.generate_query_embedding(query)

        best_match = db_handler.get_best_match(query_embedding)

        answer = best_match["documents"][0][0]

        if best_match:
            print(f"{answer}")
        else:
            print("Sorry I couldn't find a relevant answer")


if __name__ == "__main__":
    doc_paths = ["docs/doc-1.pdf", "docs/doc-2.pdf", "docs/doc-3.pdf"]
    main(doc_paths)
