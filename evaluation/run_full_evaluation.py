
import json
from pathlib import Path

from ingestion.embedding import EmbeddingService
from vectorstore.qdrant_store import QdrantVectorStore
from retrieval.retriever import Retriever

from generation.llm_client import OllamaLLM
from generation.rag_chain import RAGChain

from evaluation.evaluator import evaluate_retrieval
from evaluation.answer_evaluator import AnswerEvaluator


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"
RESULT_PATH = BASE_DIR / "full_evaluation_results.json"

TOP_K = 5


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    for index, item in enumerate(dataset, start=1):
        if not item.get("relevant_chunk_ids"):
            raise ValueError(
                f"Question {index} has no relevant chunk IDs."
            )

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore()

    try:

        if not vector_store.collection_exists():
            raise RuntimeError("Qdrant collection does not exist.")

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        llm = OllamaLLM(model="llama3.2:3b")

        rag = RAGChain(
            retriever=retriever,
            llm=llm
        )

        answer_evaluator = AnswerEvaluator(llm=llm)

        results = []

        for index, item in enumerate(dataset, start=1):

            question = item["question"]
            document_id = item["document_id"]
            relevant_ids = item["relevant_chunk_ids"]

            print(f"\nEvaluating question {index}: {question}")

            # Generate the RAG answer.
            rag_result = rag.ask(
                query=question,
                top_k=TOP_K,
                document_id=document_id
            )

            retrieved_ids = [
                chunk.chunk_id
                for chunk in rag_result.retrieved_chunks
            ]

            # Evaluate retrieval.
            retrieval_metrics = evaluate_retrieval(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=TOP_K
            )

            # Evaluate generated answer.
            if rag_result.retrieved_chunks:
                answer_metrics = answer_evaluator.evaluate(
                    question=question,
                    answer=rag_result.answer,
                    chunks=rag_result.retrieved_chunks
                )

                answer_scores = {
                    "faithfulness_score":
                        answer_metrics.faithfulness_score,
                    "answer_relevancy_score":
                        answer_metrics.answer_relevancy_score,
                    "explanation":
                        answer_metrics.explanation
                }
            else:
                answer_scores = {
                    "faithfulness_score": 0.0,
                    "answer_relevancy_score": 0.0,
                    "explanation": "No context was retrieved."
                }

            result = {
                "question": question,
                "expected_answer": item.get("expected_answer"),
                "generated_answer": rag_result.answer,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieval_metrics": retrieval_metrics,
                "answer_metrics": answer_scores
            }

            results.append(result)

            print("\nGenerated answer:")
            print(rag_result.answer)

            print("\nRetrieval metrics:")
            print(retrieval_metrics)

            print("\nAnswer metrics:")
            print(answer_scores)

        # Calculate average metrics.
        metric_names = [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "hit_rate_at_k"
        ]

        retrieval_averages = {
            metric: sum(
                result["retrieval_metrics"][metric]
                for result in results
            ) / len(results)
            for metric in metric_names
        }

        answer_averages = {
            metric: sum(
                result["answer_metrics"][metric]
                for result in results
            ) / len(results)
            for metric in [
                "faithfulness_score",
                "answer_relevancy_score"
            ]
        }

        report = {
            "total_questions": len(results),
            "top_k": TOP_K,
            "average_retrieval_metrics": retrieval_averages,
            "average_answer_metrics": answer_averages,
            "results": results
        }

        with open(RESULT_PATH, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("\n========== FINAL RAG EVALUATION ==========")

        print("\nRetrieval evaluation:")
        for metric, value in retrieval_averages.items():
            print(f"{metric}: {value:.4f}")

        print("\nAnswer evaluation:")
        for metric, value in answer_averages.items():
            print(f"{metric}: {value:.4f}")

        print(f"\nReport saved to: {RESULT_PATH}")

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
