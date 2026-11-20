
from ingestion.embedding import EmbeddingService


def main():
    service = EmbeddingService()

    text = "Docker is a containerization platform."

    embedding = service.embed_document(text)

    print("Embedding dimensions:", len(embedding))
    print("First 10 values:", embedding[:10])

    query = "What is Docker?"

    query_embedding = service.embed_query(query)

    print("Query dimensions:", len(query_embedding))


if __name__ == "__main__":
    main()
