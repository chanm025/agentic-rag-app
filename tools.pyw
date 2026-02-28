from crewai.tools import tool
from tavily import TavilyClient
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def tavily_tool(query: str) -> str:
    """Search the web for up-to-date information."""
    response = tavily_client.search(query=query, search_depth="advanced")
    return str(response)

@tool
def rag_pdf_search(query: str) -> str:
    """
    Search internal PDF knowledge base.
    Replace with FAISS + sentence-transformers implementation later.
    """
    return "PDF search not yet implemented. (Stub response)"
