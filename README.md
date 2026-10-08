# AI Research Assistant (agentic RAG)

A Streamlit app where you enter a research topic and a crew of three AI agents produces a report:

```
topic -> Researcher -> Writer -> Reviewer -> final report
          |  tools:
          |  - rag_pdf_search : FAISS search over your own PDFs
          |  - tavily_tool    : live web search (Tavily)
```

- **Researcher** searches the local PDF knowledge base and the web and writes structured notes with sources.
- **Writer** turns the notes into a readable markdown report.
- **Reviewer** critiques the report and returns a revised version.

## Retrieval pipeline (`rag.py`)
PDFs in `./data` are loaded with `PyPDFDirectoryLoader`, split into 1,000-character chunks (200 overlap), embedded locally with
`sentence-transformers/all-MiniLM-L6-v2` and indexed with FAISS. The researcher's search tool returns the top matches
labelled with file name and page number. The index is built once per run, so the first query is slower
(the embedding model, ~90 MB, is also downloaded on first use).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `crew_setup.py` | Defines the three agents, their tasks and the crew |
| `tools.py` | PDF-search and web-search tools |
| `rag.py` | PDF loading, chunking, embeddings, FAISS index, search |
| `llm_adapter.py` | LLM configuration (GPT-4o-mini via CrewAI) |
| `analysis.ipynb` | Development notebook: RetrievalQA with Llama 3.1 (Groq), embedding visualisation (PCA), knowledge-base expansion, extra agents |
| `Dockerfile`, `docker-compose.yml` | Container setup |

## Run it
1. Copy `.env.example` to `.env` and add your OpenAI and Tavily keys (never commit `.env`).
2. Put PDF files in `./data` (none are included in this repo).
3. Start the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
or with Docker: `docker compose up --build`, then open http://localhost:8501.

## Limitations
- Output quality has not been evaluated: there is no retrieval or answer-quality metric.
- The Reviewer is another LLM pass, not a fact-check against the sources, so errors can survive; check citations before relying on a report.
- Only the Researcher retrieves; the Writer and Reviewer see its notes but not the raw sources.
- A run that exceeds the timeout stops being awaited but keeps running in the background.
