"""Tools available to agents in the multi-agent RAG system."""

from langchain_core.tools import tool

from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks_with_ids



@tool(response_format="content_and_artifact")
def retrieval_tool(query: str):
    """Search the vector database for relevant document chunks.

    This tool retrieves the top 4 most relevant chunks from the Pinecone
    vector store based on the query. The chunks are formatted with page
    numbers and indices for easy reference.

    Args:
        query: The search query string to find relevant document chunks.

    Returns:
        Tuple of (serialized_content, artifact) where:
        - serialized_content: A formatted string containing the retrieved chunks
          with metadata. Format: "Chunk 1 (page=X): ...\n\nChunk 2 (page=Y): ..."
        - artifact: List of Document objects with full metadata for reference
    """
    # Retrieve documents from vector store
    docs = retrieve(query, k=4)

    # Serialize chunks with stable IDs and citations
    context, citations = serialize_chunks_with_ids(docs)

    # Return context as content and include citations in artifact
    return context, {
        "docs": docs,
        "citations": citations,
    }

