
from dataclasses import dataclass

from ingestion.models import RetrievedChunk
from retrieval.retriever import Retriever
from generation.llm_client import OllamaLLM
from generation.prompt_builder import build_prompt


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: list[RetrievedChunk]


class RAGChain:

    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM
    ):
        self.retriever = retriever
        self.llm = llm

    def ask(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None
    ) -> RAGResponse:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            document_id=document_id
        )

        # Step 2: Handle missing context
        if not chunks:
            return RAGResponse(
                answer=(
                    "I could not find relevant information "
                    "in the available documents."
                ),
                retrieved_chunks=[]
            )

        # Step 3: Build the prompt
        prompt = build_prompt(
            query=query.strip(),
            chunks=chunks
        )

        # Step 4: Generate the answer
        answer = self.llm.generate(
            prompt=prompt,
            temperature=0.2
        )

        # Step 5: Return answer and sources
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks
        )
