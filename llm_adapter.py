from dotenv import load_dotenv
from crewai import LLM

load_dotenv()  # reads OPENAI_API_KEY / TAVILY_API_KEY from a local .env file

llm = LLM(model="gpt-4o-mini", temperature=0.2)
