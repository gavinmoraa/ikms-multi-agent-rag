"""Utilities for serializing retrieved document chunks."""

from typing import List

from langchain_core.documents import Document


def serialize_chunks(docs: List[Document]) -> str:
    """Serialize a list of Document objects into a formatted CONTEXT string.

    Formats chunks with indices and page numbers as specified in the PRD:
    - Chunks are numbered (Chunk 1, Chunk 2, etc.)
    - Page numbers are included in the format "page=X"
    - Produces a clean CONTEXT section for agent consumption

    Args:
        docs: List of Document objects with metadata.

    Returns:
        Formatted string with all chunks serialized.
    """
    context_parts = []

    for idx, doc in enumerate(docs, start=1):
        # Extract page number from metadata
        page_num = doc.metadata.get("page") or doc.metadata.get(
            "page_number", "unknown"
        )

        # Format chunk with index and page number
        chunk_header = f"Chunk {idx} (page={page_num}):"
        chunk_content = doc.page_content.strip()

        context_parts.append(f"{chunk_header}\n{chunk_content}")

    return "\n\n".join(context_parts)



def serialize_chunks_with_ids(
    docs: List[Document],
) -> tuple[str, dict]:
    """
    Serialize documents into a context string with stable chunk IDs
    and a citation map for evidence-aware answers.

    Returns:
        context (str): Formatted context with chunk IDs like [C1]
        citations (dict): Chunk ID -> metadata mapping
    """
    context_parts = []
    citations = {}

    for idx, doc in enumerate(docs, start=1):
        chunk_id = f"C{idx}"

        page_num = doc.metadata.get("page") or doc.metadata.get(
            "page_number", "unknown"
        )
        source = doc.metadata.get("source", "unknown")

        chunk_content = doc.page_content.strip()

        # Context formatting with stable ID
        context_parts.append(
            f"[{chunk_id}] Chunk from page {page_num}:\n{chunk_content}"
        )

        # Citation metadata
        citations[chunk_id] = {
            "page": page_num,
            "source": source,
            "snippet": chunk_content[:100] + "..."
        }

    return "\n\n".join(context_parts), citations
