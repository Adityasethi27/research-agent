# Research Agent — Agentic AI System

A LangGraph-based research agent that autonomously decides, at each step, whether to search the web or query a local document store — then wraps it all behind a FastAPI endpoint so it can be called over HTTP.

---

## What it does

Most LLM wrappers are pipelines: input → LLM → output. This is an agent: it reasons about what tool to use, uses it, reads the result, and decides whether to answer or keep going.

Given a question, the agent:

1. Passes the full conversation history to a Gemini LLM with tools bound to it
2. If the LLM decides it needs live information → calls **TavilySearch** (web search)
3. If the LLM decides the answer is in the uploaded documents → calls **search_documents** (RAG over ChromaDB)
4. If the LLM has enough to answer → responds directly
5. The loop repeats until the agent stops calling tools

The FastAPI layer exposes this as a single `POST /ask` endpoint that takes a question and returns the agent's answer as JSON.

---

## Architecture

```
POST /ask
    │
    ▼
FastAPI endpoint (main.py)
    │
    ▼
LangGraph StateGraph (research_agent.py)
    │
    ├── chat_node: calls Gemini with full message history
    │       │
    │       └── if tool_calls in response → route to tools node
    │           else → END (return answer)
    │
    └── tools node (ToolNode)
            │
            ├── TavilySearch → live web results
            └── search_documents → ChromaDB similarity search (RAG)
```

**Key patterns:**
- `Annotated[list, add_messages]` reducer — appends each new message to state rather than replacing it, preserving conversation history across tool calls
- `bind_tools([search_tool, search_documents])` — attaches tool schemas to the LLM so it can decide which to invoke
- `ToolNode` — executes tool calls returned by the LLM and feeds results back as `ToolMessage`s
- Conditional routing via `should_use_tool` — checks the last message for `tool_calls` and routes accordingly

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (StateGraph, ToolNode, add_messages) |
| LLM | Google Gemini 2.5 Flash via LangChain |
| Web search tool | Tavily Search API |
| Document retrieval | ChromaDB + Gemini Embeddings (`models/gemini-embedding-001`) |
| API framework | FastAPI + Uvicorn |
| Secrets | python-dotenv |

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/Adityasethi27/research-agent.git
cd research-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install langgraph langchain-google-genai langchain-tavily langchain-chroma \
            langchain-core fastapi uvicorn python-dotenv chromadb
```

**3. Set up environment variables**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys:
- Gemini: [aistudio.google.com](https://aistudio.google.com) → API keys
- Tavily: [app.tavily.com](https://app.tavily.com) → API keys

**4. (Optional) Set up the RAG document store**

The `search_documents` tool queries a ChromaDB store. Point it at an existing store by updating the `persist_directory` path in `research_agent.py`, or build a fresh one using a `build_db.py` script from a compatible RAG pipeline (e.g. [pdf-chat](https://github.com/Adityasethi27/pdf-chat)).

**5. Run the server**

```bash
uvicorn main:app --reload
```

Server starts at `http://127.0.0.1:8000`

---

## Usage

**Interactive docs (easiest)**

Go to `http://127.0.0.1:8000/docs` — FastAPI auto-generates a Swagger UI. Click `/ask` → Try it out → execute with your question.

**curl**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest developments in AI agents?"}'
```

**Response**

```json
{
  "answer": "..."
}
```

**Run directly (without the API)**

```bash
python research_agent.py
```

Runs the agent against a hardcoded test question and prints the result.

---

## Project Structure

```
research-agent/
├── research_agent.py   # LangGraph agent: state, nodes, tools, graph
├── main.py             # FastAPI app: schemas, /ask endpoint
├── .env                # API keys (not committed)
├── .gitignore
└── venv/
```

---

## Related Projects

- [pdf-chat](https://github.com/Adityasethi27/pdf-chat) — RAG pipeline this agent's document tool is built on
- [ai-text-cleaner](https://github.com/Adityasethi27/ai-text-cleaner) — LangChain structured output with retry logic
- [movie-api](https://github.com/Adityasethi27/movie-api) — FastAPI + SQLAlchemy REST backend
