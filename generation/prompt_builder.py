
from ingestion.models import RetrievedChunk


def build_prompt(
    query: str,
    chunks: list[RetrievedChunk]
) -> str:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        raise ValueError("No context chunks provided.")

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        context_parts.append(
            f"""
Context {index}:
Source: {chunk.source}
Page: {chunk.page_number}
Content:
{chunk.text}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful assistant that answers questions
using the provided document context.

Instructions:
1. Answer using only the provided context.
2. Do not invent facts or information.
3. If the answer is not available in the context,
   clearly say that the document does not provide
   enough information.
4. Give clear and understandable answers.
5. Mention relevant page numbers when possible.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    return prompt.strip()
