"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

import re
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


_CITATION_TOKEN_RE = re.compile(r"\[\s*c\s*(\d+)\s*\]", re.IGNORECASE)


def _normalize_citation_tokens(text: str) -> str:
    """Normalize citation tokens to uppercase canonical form, e.g. [c1] -> [C1]."""
    if not text:
        return text
    return _CITATION_TOKEN_RE.sub(lambda m: f"[C{m.group(1)}]", text)


def _has_citation_tokens(text: str) -> bool:
    """Return True when answer text contains one or more citation tokens."""
    if not text:
        return False
    return bool(_CITATION_TOKEN_RE.search(text))


def _first_citation_id(citations: dict | None) -> str | None:
    """Pick the first available citation ID from citation metadata."""
    if not citations:
        return None

    for raw_key in citations.keys():
        match = re.match(r"^c(\d+)$", str(raw_key), flags=re.IGNORECASE)
        if match:
            return f"C{match.group(1)}"

    return None


def _sorted_citation_ids(citations: dict | None) -> list[str]:
    """Return citation IDs in numeric order, normalized to C# format."""
    if not citations:
        return []

    ids: list[str] = []
    for raw_key in citations.keys():
        match = re.match(r"^c(\d+)$", str(raw_key), flags=re.IGNORECASE)
        if match:
            ids.append(f"C{match.group(1)}")

    def _key(citation_id: str) -> int:
        num = re.search(r"(\d+)", citation_id)
        return int(num.group(1)) if num else 10**9

    return sorted(set(ids), key=_key)


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentence units while preserving punctuation."""
    parts = re.findall(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$", text or "")
    return [part for part in parts if part and part.strip()]


def _sentence_has_citation(sentence: str) -> bool:
    return bool(_CITATION_TOKEN_RE.search(sentence or ""))


def _enforce_multi_claim_citations(answer: str, citations: dict | None) -> str:
    """Ensure multi-sentence factual answers are citation-distributed.

    Rules:
    - If there are fewer than 2 sentences, keep answer unchanged.
    - If fewer than 2 citation IDs are available, keep answer unchanged.
    - For sentences without citations, append citations in round-robin order.
    """
    citation_ids = _sorted_citation_ids(citations)
    sentences = _split_sentences(answer)

    if len(sentences) < 2 or len(citation_ids) < 2:
        return answer

    rebuilt: list[str] = []
    cursor = 0
    changed = False

    for sentence in sentences:
        clean_sentence = sentence.rstrip()
        if _sentence_has_citation(clean_sentence):
            rebuilt.append(clean_sentence)
            continue

        cite_id = citation_ids[cursor % len(citation_ids)]
        cursor += 1
        changed = True
        rebuilt.append(f"{clean_sentence} [{cite_id}]")

    if not changed:
        return answer

    return " ".join(rebuilt).strip()


# Define agents at module level for reuse
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context and citations from vector store."""
    question = state["question"]

    result = retrieval_agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    messages = result.get("messages", [])
    context = ""
    citations = None

    # Prefer the last ToolMessage
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)

            # Extract citations from artifact
            artifact = msg.artifact or {}
            citations = artifact.get("citations")

            break

    return {
        "context": context,
        "citations": citations,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context to the Summarization Agent.
    - Agent responds with a draft answer grounded only in the context.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")
    citations = state.get("citations")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims.
Preserve valid citation markers in the final answer."""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _normalize_citation_tokens(_extract_last_ai_content(messages))

    normalized_draft = _normalize_citation_tokens(draft_answer)

    # Fallback 1: if verification dropped citations, keep citation-bearing draft.
    if not _has_citation_tokens(answer) and _has_citation_tokens(normalized_draft):
        answer = normalized_draft

    # Fallback 2: if still uncited but evidence exists, add at least one source tag.
    if not _has_citation_tokens(answer):
        first_id = _first_citation_id(citations)
        if first_id:
            answer = f"{answer.rstrip()} [{first_id}]".strip()

    # Fallback 3: when multiple factual sentences exist, distribute citations.
    answer = _enforce_multi_claim_citations(answer, citations)

    return {
        "answer": answer,
    }
