from pathlib import Path
import re

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .models import QuestionRequest, QAResponse
from .services.qa_service import answer_question
from .services.indexing_service import index_pdf_file

_CITATION_TOKEN_RE = re.compile(r"\[\s*c\s*(\d+)\s*\]", re.IGNORECASE)


def _normalize_answer_citations(answer: str) -> str:
    if not answer:
        return answer
    return _CITATION_TOKEN_RE.sub(lambda m: f"[C{m.group(1)}]", answer)


def _normalize_citation_map(citations: dict | None) -> dict | None:
    if not isinstance(citations, dict):
        return citations

    normalized: dict[str, dict] = {}
    for raw_key, value in citations.items():
        key_str = str(raw_key)
        match = re.match(r"^c(\d+)$", key_str, flags=re.IGNORECASE)
        key = f"C{match.group(1)}" if match else key_str
        normalized[key] = value

    return normalized


app = FastAPI(
    title="Class 12 Multi-Agent RAG Demo",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ikms-multi-agent-lxcoxjm9x-gavin-moragodas-projects.vercel.app",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict:
    """Basic health/info route for quick browser checks."""

    return {
        "status": "ok",
        "message": "IKMS Multi-Agent RAG API is running.",
        "docs": "/docs",
    }

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover - simple demo handler
    """Catch-all handler for unexpected errors.

    FastAPI will still handle `HTTPException` instances and validation errors
    separately; this is only for truly unexpected failures so API consumers
    get a consistent 500 response body.
    """

    if isinstance(exc, HTTPException):
        # Let FastAPI handle HTTPException as usual.
        raise exc

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    """Submit a question about the vector databases paper.

    US-001 requirements:
    - Accept POST requests at `/qa` with JSON body containing a `question` field
    - Validate the request format and return 400 for invalid requests
    - Return 200 with `answer`, `draft_answer`, and `context` fields
    - Delegate to the multi-agent RAG service layer for processing
    """

    question = payload.question.strip()
    if not question:
        # Explicit validation beyond Pydantic's type checking to ensure
        # non-empty questions.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    # Delegate to the service layer which runs the multi-agent QA graph
    result = answer_question(question)

    normalized_answer = _normalize_answer_citations(result.get("answer", ""))
    normalized_citations = _normalize_citation_map(result.get("citations"))

    return QAResponse(
        answer=normalized_answer,
        context=result.get("context", ""),
        citations=normalized_citations,
    )


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    """Upload a PDF and index it into the vector database.

    This endpoint:
    - Accepts a PDF file upload
    - Saves it to the local `data/uploads/` directory
    - Uses PyPDFLoader to load the document into LangChain `Document` objects
    - Indexes those documents into the configured Pinecone vector store
    """

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    # Index the saved PDF
    chunks_indexed = index_pdf_file(file_path)

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
        "message": "PDF indexed successfully.",
    }
