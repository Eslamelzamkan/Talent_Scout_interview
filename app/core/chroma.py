from __future__ import annotations

from typing import Any

from chromadb.api import AsyncClientAPI

from app.core.llm import LLMClient
from app.models import RubricModel

_chroma: AsyncClientAPI | None = None
_llm: LLMClient | None = None


def init(chroma: AsyncClientAPI, llm: LLMClient) -> None:
    global _chroma, _llm
    _chroma = chroma
    _llm = llm


def _require_clients() -> tuple[AsyncClientAPI, LLMClient]:
    if _chroma is None or _llm is None:
        raise RuntimeError("chroma module is not initialized")
    return _chroma, _llm


def _collection_name(job_id: str) -> str:
    return f"questions_{job_id}"


async def seed(job_id: str, questions: list[dict[str, Any]], *, replace: bool = True) -> int:
    if not questions:
        return 0
    chroma, llm = _require_clients()
    texts = [str(question["text"]) for question in questions]
    embeddings = await llm.embed(texts)
    if replace:
        try:
            await chroma.delete_collection(_collection_name(job_id))
        except Exception:
            pass
    collection = await chroma.get_or_create_collection(_collection_name(job_id))
    ids = [f"{job_id}:{index}" for index in range(len(questions))]
    metadatas = [
        {
            "dimension": str(question["dimension"]),
            "seniority": str(question["seniority"]),
            "topic": str(question["topic"]),
            "lane": str(question.get("lane") or "project_deep_dive"),
        }
        for question in questions
    ]
    await collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(questions)


async def retrieve(
    job_id: str,
    dimension: str,
    seniority: str,
    lane: str | None = None,
    n: int = 3,
    exclude: list[str] | None = None,
) -> list[str]:
    chroma, llm = _require_clients()
    collection = await chroma.get_or_create_collection(_collection_name(job_id))
    query_embedding = (await llm.embed([dimension]))[0]
    excluded = set(exclude or [])
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": max(n + len(excluded), n),
        "include": ["documents"],
    }
    where_conditions: list[dict[str, str]] = [
        {"dimension": dimension},
        {"seniority": seniority},
    ]
    if lane:
        where_conditions.append({"lane": lane})
    where_filter = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
    results = await collection.query(
        where=where_filter,
        **query_kwargs,
    )
    documents = results.get("documents", [[]])
    if lane and not documents[0]:
        results = await collection.query(
            where={"$and": [{"dimension": dimension}, {"seniority": seniority}]},
            **query_kwargs,
        )
        documents = results.get("documents", [[]])
    filtered: list[str] = []
    seen: set[str] = set()
    for item in documents[0]:
        if not item or item in excluded or item in seen:
            continue
        filtered.append(item)
        seen.add(item)
        if len(filtered) == n:
            break
    return filtered


async def seed_question_bank(job_id: str, rubric: RubricModel) -> None:
    questions = [
        {
            "text": question,
            "dimension": dimension.name,
            "seniority": rubric.role_level,
            "topic": dimension.name,
            "lane": "project_deep_dive",
        }
        for dimension in rubric.dimensions
        for question in (dimension.sample_questions or [f"Tell me about {dimension.name}."])
    ]
    await seed(job_id, questions)


async def retrieve_questions(job_id: str, topic: str, k: int = 5) -> list[str]:
    chroma, llm = _require_clients()
    collection = await chroma.get_or_create_collection(_collection_name(job_id))
    query_embedding = (await llm.embed([topic]))[0]
    results = await collection.query(
        query_embeddings=[query_embedding],
        where={"dimension": topic},
        n_results=k,
        include=["documents"],
    )
    return [item for item in results.get("documents", [[]])[0] if item]
