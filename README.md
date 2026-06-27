# AI Codebase Assistant

A localized, multi-agent RAG tool designed to parse, index, and answer structural questions about public GitHub repositories. Built with a modular Python backend (FastAPI + LangGraph) and an interactive, clean vanilla frontend.

---

## System Architecture

The application is split into two distinct execution phases to decouple data processing from cognitive analysis:

### Phase 1: In-Memory Ingestion Pipeline
1. **Repository Ingestion:** Uses `GitPython` to programmatically clone target repositories to a local temporary folder.
2. **Structural Text Chunking:** Sequentially crawls source code files (e.g., `.java`, `.py`), applying a line-bound sliding window chunking mechanism (25-line window with a 5-line overlap).
3. **Local Vectorization:** Runs text blocks through a local `Sentence-Transformers` model (`BAAI/bge-small-en-v1.5`) to output dense 384-dimensional semantic embeddings.
4. **Transient Vector Storage:** Vectors and code metadata are stored inside an in-memory (`:memory:`) instance of `Qdrant` for fast spatial similarity calculations.

### Phase 2: Stateful Multi-Agent Orchestration
Once indexed, user questions route through a state-driven pipeline compiled with **LangGraph**:
* **Planner Node:** Uses `Gemini 2.5 Flash` to extract programmatic intent from the query and create a search keyword plan.
* **Retrieval Node:** A deterministic function that executes a vector similarity search across Qdrant using the calculated plan to fetch the top 3 relevant chunks.
* **Reasoning Node:** Leverages `Gemini 2.5 Flash` to digest raw source code chunks and formulate a technical breakdown.
* **Citation Node:** Programmatically appends original file paths and line ranges from vector metadata, preventing LLM citation hallucination.

---

## Tech Stack

### Backend Engine
* **Framework:** FastAPI (Asynchronous Python API Layer)
* **Agent Orchestration:** LangGraph & LangChain (State-machine workflow execution)
* **LLM Provider:** Google Gemini API (`gemini-2.5-flash`)
* **Vector Search Engine:** Qdrant DB (Configured as a localized in-memory collection)
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (via Sentence-Transformers)
* **File Operations:** GitPython

### Frontend Interface
* **Structure & UI:** HTML5, Vanilla JavaScript (ES6+), Bootstrap 5
* **Markdown Parser:** Lightweight regex token replacement helper

---

## Setup

1. Create a virtual environment `python -m venv venv`.
2. Install all the dependecies from requirements.txt `pip install -r requirements.txt`.
3. Add your Gemini API key in core/graph.py.
4. Run the FastAPI Server `uvicorn app:main --reload`.
5. Make the Frontend Server live.
6. Input any GitHub repository and get its codebase assistance through LLM.

