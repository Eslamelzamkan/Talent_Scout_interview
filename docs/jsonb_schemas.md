# JSONB Schema Versions

These fields are intentionally schemaless in Postgres and validated in application code before write and after read:

- `interview_sessions.candidate_context`
- `parsed_job_contexts.rubric`
- `scorecards.interview_dimension_scores`
- `scorecards.judge_ensemble_raw`

## Current Versions

### `candidate_context`

- Version: `v1`
- Owner model: `CandidateContext`
- Storage shape: object with optional summary and highlight lists used to ground interview questions

### `rubric`

- Version: `v1`
- Owner model: `RubricDimension`
- Storage shape: list of rubric-dimension objects

### `interview_dimension_scores`

- Version: `v1`
- Owner model: `DimensionScore`
- Storage shape: object keyed by dimension name

### `judge_ensemble_raw`

- Version: `v1`
- Owner model: `JudgeEnsembleRaw`
- Storage shape: judge list with nested per-dimension votes

## Evolution Rules

1. Add new fields as optional in the owning Pydantic model with a safe default.
2. Write a backfill script named `scripts/migrate_jsonb_<field>_<version>.py`.
3. Read every existing row, validate through the latest model, and rewrite the normalized payload.
4. Do not rely on Alembic autogenerate for JSONB changes. It will not detect application-level shape changes.
5. Record every schema version change in this document before deploying the migration.
