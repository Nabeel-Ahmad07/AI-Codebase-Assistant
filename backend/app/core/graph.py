import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

GEMINI_API_KEY = "INSERT YOUR GEMINI API KEY HERE"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

class AgentState(TypedDict):
    user_query: str
    plan: str
    retrieved_chunks: List[dict]
    analysis: str
    final_response: str


def planner_agent(state: AgentState):
    query = state["user_query"]
    system_prompt = (
        "You are an AI Software Architect. The user is asking about a codebase.\n"
        "Create a short, targeted search strategy or keyword list to find relevant "
        "functions, classes, or network structures in a vector database."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Question: {query}")
    ])
    return {"plan": response.content}


def retrieval_agent(state: AgentState):
    from app.main import db_service 
    
    search_query = f"{state['user_query']} {state['plan']}"
    chunks = db_service.search_similar_code(query=search_query, limit=3)
    return {"retrieved_chunks": chunks}


def reasoning_agent(state: AgentState):
    query = state["user_query"]
    chunks = state["retrieved_chunks"]
    
    code_context = ""
    for idx, c in enumerate(chunks):
        code_context += f"--- Code Snippet #{idx+1} ({c['metadata']['file_path']}) ---\n{c['content']}\n\n"

    system_prompt = (
        "You are an expert Developer Engine. Look at the retrieved code blocks "
        "and explain exactly how they work to answer the user's question. "
        "Focus purely on the logic present in the code. Do not make up files."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {query}\n\nCode Context:\n{code_context}")
    ])
    return {"analysis": response.content}


def citation_agent(state: AgentState):
    analysis = state["analysis"]
    chunks = state["retrieved_chunks"]
    
    citation_summary = "\n\n** Source References:**\n"
    seen_references = set()
    
    for c in chunks:
        ref_str = f"- `{c['metadata']['file_path']}` (Lines {c['metadata']['start_line']}-{c['metadata']['end_line']})"
        if ref_str not in seen_references:
            seen_references.add(ref_str)
            citation_summary += ref_str + "\n"
        
    return {"final_response": f"{analysis}{citation_summary}"}

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_agent)
workflow.add_node("retriever", retrieval_agent)
workflow.add_node("reasoner", reasoning_agent)
workflow.add_node("citation", citation_agent)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "reasoner")
workflow.add_edge("reasoner", "citation")
workflow.add_edge("citation", END)

graph_engine = workflow.compile()