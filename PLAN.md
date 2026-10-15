# RAG Document Q&A: Project Plan

## Goal
Build a simple document RAG system (retrieval-augmented generation) that answers questions about my own documents, exposed through a REST API and runnable with Docker.

The project exists to practice: **REST API, Git, Docker, LlamaIndex (or LangChain), and building a RAG pipeline end to end.** Keep it simple. Prefer fewer moving parts over clever architecture.

Reference project (the starting point, CLI only): https://nextwork.ai/projects/02fbff81-6f71-4e6d-b2a0-90b9dd43818f
It uses Qdrant in Docker Compose, LlamaIndex for ingest/query scripts, and an optional Streamlit UI. This project extends it with a FastAPI REST layer, a containerized app, and a proper Git workflow.

## Stack
| Part | Choice |
|---|---|
| Language | Python 3.12 |
| RAG framework | LlamaIndex (LangChain only if there is a clear reason later) |
| Vector DB | Qdrant (Docker, persistent volume) |
| API | FastAPI + Uvicorn |
| Containers | Docker + Docker Compose (`qdrant` and `api` services) |
| Version control | Git, small commits per milestone |
| LLM + embeddings | **TBD: decide before Milestone 3** (OpenAI, Anthropic, or local via Ollama). Keep it swappable through `config.py` and `.env`. |

## Scope
**In scope (core):** text documents (`.txt`, `.md`, `.pdf`): ingest, chunk, embed, store, retrieve, answer with sources.

**Out of scope until core is done (stretch):** websites, external APIs, images.

## Target layout
```
rag-project/
├── app/
│   ├── main.py        # FastAPI app and routes
│   ├── ingest.py      # load, chunk, embed, store
│   ├── query.py       # retrieve and generate answer
│   └── config.py      # settings loaded from environment
├── data/              # sample documents (gitignored except a sample file)
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example       # variable names only, never real keys
├── .gitignore
├── PLAN.md
└── README.md
```

## Milestones
Work on **one milestone at a time**. Each ends with a working, testable result and a Git commit.

### M1: Repo setup
- [x] `git init`, `.gitignore` (`.env`, `venv/`, `__pycache__/`, `data/*` except sample)
- [x] Create folder layout above, `requirements.txt`, `.env.example`
- [x] README skeleton
- **Done when:** repo has a clean first commit and no secrets are tracked.

### M2: Qdrant in Docker
- [ ] `docker-compose.yml` with `qdrant` service, pinned image version, ports 6333/6334, named volume
- **Done when:** `docker compose up -d` starts it and the dashboard loads at `http://localhost:6333/dashboard`.

### M3: Ingestion pipeline
- [ ] `config.py` reads settings from env (API keys, Qdrant host, collection name, chunk size)
- [ ] `ingest.py` loads files from `data/`, splits with `SentenceSplitter` (start with chunk_size=512, overlap=50), embeds, stores in Qdrant
- [ ] Add a sample document in `data/`
- **Done when:** running the script creates a collection with points visible in the dashboard.

### M4: Query engine
- [ ] `query.py` connects to the existing collection, retrieves top-k chunks (start k=3), generates an answer
- [ ] Return source file names and similarity scores with each answer
- **Done when:** a question about the sample doc returns a grounded answer plus sources.

### M5: REST API
- [ ] `GET /health` returns service status
- [ ] `POST /documents` uploads a file and ingests it
- [ ] `POST /query` takes `{"question": "..."}` and returns `{"answer": "...", "sources": [{"file": "...", "score": 0.0}]}`
- [ ] Basic error handling (empty question, unsupported file type, Qdrant unreachable)
- **Done when:** all endpoints work from `http://localhost:8000/docs`.

### M6: Dockerize the API
- [ ] `Dockerfile` for the API
- [ ] Add `api` service to Compose; it reaches Qdrant by service name (`qdrant`), not `localhost`
- [ ] Pass secrets through `.env`, never bake them into the image
- **Done when:** `docker compose up --build` on a fresh clone gives a working API.

### M7: Polish
- [ ] A few tests (e.g. chunking, `/health`, `/query` with a mocked LLM)
- [ ] README: what it is, how to run, example `curl` commands
- [ ] Tag `v0.1.0`
- **Done when:** someone else could clone the repo and run it from the README alone.

## Stretch milestones (only after M7)
- **S1: Websites:** `POST /ingest/url` fetches a page, extracts text, ingests it.
- **S2: External APIs:** fetch JSON from a public API, convert to text, ingest it.
- **S3: Images:** caption with a vision model (or use CLIP embeddings), store in a second collection, and let `/query` search both.

## Working agreement for Claude Code
- Read this file at the start of each session and tell me which milestone we are on.
- Do only the current milestone. Do not jump ahead or add features not listed here.
- Explain what each file does in a sentence or two so I learn as we build.
- Keep code simple and commented where the logic is not obvious.
- Never commit secrets. Use `.env` and keep `.env.example` current.
- Do not commit after each milestone. I will commit the file after reviewing.
- When a choice has trade-offs (library, chunk size, model), give me the options briefly and recommend one.
- Check off completed items in this file as we go.

## Open decisions
- [ ] LLM and embedding provider (affects cost and config)
- [ ] Whether to add a simple UI (Streamlit) after M7 or go straight to stretch goals
