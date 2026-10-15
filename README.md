# RAG Document Q&A

A simple retrieval-augmented generation (RAG) system that answers questions about your own documents (`.txt`, `.md`, `.pdf`), with sources, through a REST API and a small demo web page.

**Stack:** Python 3.12, LlamaIndex, Qdrant, FastAPI, Docker Compose. DeepSeek writes the answers; a local Ollama model (`nomic-embed-text`) creates the embeddings.

## How it works
1. **Ingest:** files in `data/` are split into chunks (512 tokens, 50 overlap), embedded, and stored in Qdrant.
2. **Query:** the question is embedded, the 3 most similar chunks are retrieved, and DeepSeek answers using only those chunks.
3. The answer comes back with the source file names and similarity scores.

## Prerequisites
- Docker (with Compose)
- [Ollama](https://ollama.com) running on the host, with the embedding model pulled: `ollama pull nomic-embed-text`
- A [DeepSeek API key](https://platform.deepseek.com)

## Run it
```bash
git clone https://github.com/benmanafe/multimodal-rag.git
cd multimodal-rag
cp .env.example .env          # then put your DEEPSEEK_API_KEY in .env
docker compose up --build -d
docker compose exec api python -m app.ingest   # index the sample document (data/sample.md)
```
- Demo page: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

Stop with `docker compose down` (add `-v` to also delete the stored vectors).

## API examples
```bash
# Health
curl http://localhost:8000/health

# Upload and index a document (re-indexes everything in data/)
curl -F "file=@notes.pdf" http://localhost:8000/documents

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Why does chunk size matter?"}'
# {"answer": "...", "sources": [{"file": "sample.md", "score": 0.67}]}
```
Errors: `400` unsupported file type, `422` empty question, `404` nothing ingested yet, `503` Qdrant or Ollama unreachable.

## Run without Docker for the API
```bash
pip install -r requirements.txt
docker compose up -d qdrant     # Qdrant still runs in Docker
python -m app.ingest
uvicorn app.main:app --reload
```
Outside Docker, `.env` should keep `QDRANT_HOST=localhost` and `OLLAMA_BASE_URL=http://localhost:11434`.

## Tests
```bash
pytest
```
The tests mock Qdrant and the LLM, so they need no services or API keys.

## Configuration
All settings live in `.env` (see `.env.example`): model names, Qdrant host and collection, chunk size and overlap, top-k.

## Project layout
- `app/main.py`: FastAPI routes and the demo page
- `app/ingest.py`: load, chunk, embed, store
- `app/query.py`: retrieve and generate the answer
- `app/config.py`: settings from the environment
- `app/static/index.html`: demo web page
- `data/`: documents to ingest (gitignored except the sample)
- `tests/`: tests

## Limitations
- Each upload re-indexes every document in `data/` (simple, but slow for large folders).
- No authentication; intended for local use.
