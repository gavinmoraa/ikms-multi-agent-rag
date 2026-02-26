📘 IKMS Multi-Agent RAG – Evidence-Aware QA with Citations
📌 Project Overview

This project extends the existing IKMS Multi-Agent RAG system by implementing
Feature 4: Evidence-Aware Answers with Chunk Citations.

The enhancement ensures that every generated answer is:

Grounded in retrieved documents

Accompanied by explicit, traceable citations

Fully transparent and verifiable by users

The system is built using LangChain v1, LangGraph, FastAPI, and Pinecone.

🎯 Selected Feature
Feature 4: Evidence-Aware Answers with Chunk Citations

Objective:
Enhance the RAG pipeline so that answers include inline citations (e.g. [C1], [C2]) that map directly to retrieved document chunks and page numbers.

🧠 Motivation

In the original pipeline:

Answers were generated as plain text

Users could not verify where information came from

There was no mapping between answers and source documents

This feature introduces full evidence transparency, which is essential for:

Trustworthy AI systems

Academic and enterprise use cases

Debugging and evaluation of RAG pipelines

🏗️ Architecture Overview
🔁 Multi-Agent Flow
User Question
   ↓
Retrieval Agent (Pinecone)
   ↓
Chunk Serialization (with IDs)
   ↓
Summarization Agent (citation-aware)
   ↓
Verification Agent
   ↓
Final Answer + Citations

🔧 Key Enhancements Implemented
1️⃣ Stable Chunk Identifiers

Each retrieved document chunk is assigned a stable ID:

[C1], [C2], [C3], ...


These IDs remain consistent throughout the pipeline and are used for citation.

2️⃣ Citation Map Generation

During retrieval, a citation map is created:

{
  "C1": {
    "page": 1,
    "source": "vectorpdf.pdf",
    "snippet": "Modern deep learning models capture the semantics..."
  }
}


This map links each chunk ID to:

Page number

Source file

Text snippet


3️⃣ Citation-Aware Summarization

The Summarization Agent is instructed to:

Use only retrieved context

Add citations immediately after supported statements

Never invent citations

Example answer:

Modern deep learning models capture the semantics of complex data by transforming it into high-dimensional embeddings [C1].

4️⃣ Verification-Safe Citations

The Verification Agent:

Removes unsupported claims

Ensures citations remain valid

Prevents hallucinated evidence

5️⃣ API Response with Evidence

The /qa endpoint returns:

{
  "answer": "... [C1]",
  "context": "...",
  "citations": {
    "C1": { "page": 1, "source": "...", "snippet": "..." }
  }
}


This allows the frontend to display clickable, inspectable sources.

🖥️ User Interface

A lightweight HTML UI was implemented to demonstrate:

Question input

Generated answer

List of cited sources with page numbers

This satisfies the UI requirement while keeping the focus on backend agent behavior.

🧪 Example End-to-End Output

Question

What do modern deep learning models capture?


Answer

Modern deep learning models capture the semantics of complex data by transforming them into high-dimensional embedding vectors [C1].


Sources

[C1] Page 1 – vectorpdf.pdf

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install fastapi uvicorn langchain langgraph pinecone-client

2️⃣ Start the Backend
python -m uvicorn src.app.api:app --reload


Open:

http://127.0.0.1:8000/docs

3️⃣ Index a PDF

Use the /index-pdf endpoint to upload and index a document.

4️⃣ Ask Questions

Use /qa or the provided UI (ui/index.html) to ask questions and view cited answers.

🧠 Learning Outcomes

Through this feature, the following concepts were applied:

LangGraph state propagation

Retrieval-augmented generation (RAG)

Stable context serialization

Evidence transparency in LLM systems

Pinecone vector indexing and embedding alignment

Real-world debugging of ingestion and embedding mismatches

🏁 Conclusion

This implementation successfully transforms the IKMS RAG pipeline into an evidence-aware system, ensuring that all generated answers are transparent, verifiable, and grounded in retrieved documents.

The project meets all acceptance criteria for Feature 4 and demonstrates production-ready RAG design principles.

👤 Author

Gavin Moragoda
Multi-Agent RAG Assignment