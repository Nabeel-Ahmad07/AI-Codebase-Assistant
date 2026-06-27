from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.parser import CodeParser
from app.services.vector_db import VectorDBService
from app.core.graph import graph_engine  

app = FastAPI(
    title="AI Software Engineering Assistant",
    description="Resume Project: RAG-based codebase intelligence tool using LangGraph, FastAPI, and Qdrant."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = CodeParser()
db_service = VectorDBService()

class RepoUploadRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Software Engineering Assistant API is active."}

@app.post("/api/repository/process")
async def process_repository(payload: RepoUploadRequest):
    """
    Endpoint that accepts a public GitHub link, clones it locally,
    chunks the source files, and builds vector indexes in Qdrant.
    """
    url_str = payload.url.strip()
    if not url_str.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="Invalid URL. Please provide a valid public GitHub repository link.")

    try:
        local_repo_path = parser.clone_repository(url_str)
        chunks = parser.process_repository(local_repo_path)
        if not chunks:
            return {"success": False, "message": "No supported code files found in the repository."}

        db_service.upsert_code_chunks(chunks)
        return {
            "success": True, 
            "message": f"Successfully parsed and indexed {len(chunks)} code blocks from the repository."
        }

    except Exception as e:
        print(f"Error processing repository: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process repository: {str(e)}")

@app.post("/api/chat/placeholder")
async def chat_placeholder(payload: ChatRequest):
    """
    Routes the user message directly through the multi-agent graph system.
    """
    try:
        print(f"\n[AGENT ENGINE] Invoking multi-agent state graph for query: '{payload.message}'")
        
        initial_state = {
            "user_query": payload.message,
            "plan": "",
            "retrieved_chunks": [],
            "analysis": "",
            "final_response": ""
        }
        
        final_state = graph_engine.invoke(initial_state)
        formatted_snippets = []
        for chunk in final_state["retrieved_chunks"]:
            formatted_snippets.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            })
            
        print("[AGENT ENGINE] Execution cycle completed successfully.\n")
        return {
            "user_query": payload.message,
            "final_answer": final_state["final_response"],
            "matched_code_snippets": formatted_snippets
        }
        
    except Exception as e:
        print(f" Error executing agent workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")