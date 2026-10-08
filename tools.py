import os

from crewai.tools import tool

from rag import search_pdfs


@tool("rag_pdf_search")
def rag_pdf_search(query: str) -> str:
    """Search the local PDF knowledge base and return the most relevant passages with their sources."""
    return search_pdfs(query)


@tool("tavily_tool")
def tavily_tool(query: str) -> str:
    """Search the web for up-to-date information."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable: TAVILY_API_KEY is not set."
    from tavily import TavilyClient
    response = TavilyClient(api_key=api_key).search(query=query, search_depth="advanced")
    return str(response)
