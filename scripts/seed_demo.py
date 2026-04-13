from __future__ import annotations

import asyncio

from sqlmodel import select

from app.core.db import AsyncSessionFactory, create_tables
from app.models import InterviewSession, ParsedJobContext


async def main() -> None:
    await create_tables()
    async with AsyncSessionFactory() as session:
        existing = (
            await session.exec(
                select(ParsedJobContext).where(ParsedJobContext.job_id == "demo-backend")
            )
        ).first()
        if existing is None:
            job = ParsedJobContext(
                job_id="demo-backend",
                role_title="Senior Backend Engineer",
                seniority="Senior",
                domain="Backend",
                must_have_skills=["Python", "Postgres"],
                rubric=[
                    {
                        "name": "System Design",
                        "weight": 0.5,
                        "description": "Ability to design systems.",
                        "score_anchor_3": "Great",
                        "score_anchor_2": "Good",
                        "score_anchor_1": "Poor",
                        "few_shot_3": "Good answer",
                        "few_shot_2": "Okay answer",
                        "few_shot_1": "Bad answer",
                        "sample_questions": ["Design Twitter"]
                    },
                    {
                        "name": "Python Fundamentals",
                        "weight": 0.5,
                        "description": "Knowledge of Python.",
                        "score_anchor_3": "Great",
                        "score_anchor_2": "Good",
                        "score_anchor_1": "Poor",
                        "few_shot_3": "Good answer",
                        "few_shot_2": "Okay answer",
                        "few_shot_1": "Bad answer",
                        "sample_questions": ["What is a generator?"]
                    }
                ],
                question_seed_topics=["System Design", "Python"],
                requires_coding=True,
            )
            session.add(job)
            await session.flush()
            session.add(
                InterviewSession(
                    job_id=job.job_id,
                    candidate_id="4c9b5d26-807c-4286-8d6e-94539f41afa5",
                    candidate_name="Demo Candidate",
                    candidate_email="demo@example.com",
                    screening_score=0.92,
                    extracted_skills=["Python", "Postgres"],
                    assessment_score=0.88,
                    assessment_type="verbal technical",
                    weak_areas=["Ownership"],
                    livekit_room_name="demo-backend-demo-candidate",
                    langgraph_thread_id="demo-thread",
                    recruiter_id="demo-recruiter",
                    candidate_context={},
                )
            )
            await session.commit()


if __name__ == "__main__":
    asyncio.run(main())
