
import json
from pathlib import Path

from ingestion.embedding import EmbeddingService
from vectorstore.qdrant_store import QdrantVectorStore
from retrieval.retriever import Retriever

from evaluation.evaluator import evaluate_retrieval


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"
RESULT_PATH = BASE_DIR / "retrieval_results.json"

TOP_K = 5


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    # Validate ground-truth labels before querying Qdrant.
    for index, item in enumerate(dataset, start=1):
        if not item.get("relevant_chunk_ids"):
            raise ValueError(
                f"Question {index} has no relevant_chunk_ids. "
                "Label the relevant chunks before evaluation."
            )

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore()

    results = []

    try:

        if not vector_store.collection_exists():
            raise RuntimeError("Qdrant collection does not exist.")

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        for index, item in enumerate(dataset, start=1):

            question = item["question"]
            document_id = item["document_id"]
            relevant_ids = item["relevant_chunk_ids"]

            print(f"\nEvaluating question {index}: {question}")

            retrieved_chunks = retriever.retrieve(
                query=question,
                top_k=TOP_K,
                document_id=document_id
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk in retrieved_chunks
            ]

            metrics = evaluate_retrieval(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=TOP_K
            )

            result = {
                "question": question,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": relevant_ids,
                "metrics": metrics
            }

            results.append(result)

            print("Metrics:", metrics)

        # Calculate average metrics.
        metric_names = [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "hit_rate_at_k"
        ]

        averages = {
            metric: sum(
                result["metrics"][metric]
                for result in results
            ) / len(results)
            for metric in metric_names
        }

        report = {
            "top_k": TOP_K,
            "total_questions": len(results),
            "average_metrics": averages,
            "results": results
        }

        with open(RESULT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n========== EVALUATION SUMMARY ==========")

        for metric, value in averages.items():
            print(f"{metric}: {value:.4f}")

        print(f"\nReport saved to: {RESULT_PATH}")

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
