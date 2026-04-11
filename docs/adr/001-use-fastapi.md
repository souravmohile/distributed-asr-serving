# ADR 001: Use FastAPI for API Layer

## Context
The project needs an HTTP API to accept audio files and return transcripts of those audio files.
A decision was needed on how to build this layer.

## Options Considered
- **Raw Python (http.server / sockets)** — more granular control, no dependencies
- **FastAPI** — minimal boilerplate, handles file uploads cleanly

## Decision
Use FastAPI with uvicorn.

## Consequences
- We accept a framework dependency in exchange for faster development
- Frees up time to focus on the distributed infra layer which is the actual goal
- If we ever need raw socket-level control, we'd need to revisit this