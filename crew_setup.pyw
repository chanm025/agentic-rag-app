from crewai import Agent, Task, Crew
from llm_adapter import llm
from tools import rag_pdf_search, tavily_tool
import concurrent.futures

def run_crew(topic: str, timeout=120):

    researcher = Agent(
    role="Researcher",
    goal=(
        "Conduct deep research on a given query using both the local PDF "
        "knowledge base and real-time web search."
    ),
    backstory=(
        "You are an expert research analyst. You know how to read vectorstore "
        "results from the PDF knowledge base and combine them with fresh web "
        "information to produce a thorough evidence summary."
    ),
    tools=[], 
    llm=llm,
    verbose=True,
    )

    writer = Agent(
    role="Content Writer",
    goal=(
        "Generate clear, structured, well-organised answers or reports based "
        "on the Researcher’s findings."
    ),
    backstory=(
        "You are a skilled technical writer. You take research notes and turn "
        "them into polished, easy-to-read explanations and reports."
    ),
    tools=[],  
    llm=llm,
    verbose=True,
    )

    critic = Agent(
    role="Reviewer",
    goal=(
        "Review the Writer’s answer for factual accuracy, coherence, and "
        "completeness, and suggest improvements."
    ),
    backstory=(
        "You are a meticulous reviewer who checks arguments, corrects errors, "
        "and improves clarity and structure."
    ),
    tools=[],  
    llm=llm,
    verbose=True,
    )
    
    research_task = Task(
    description=(
        "Research the topic: {topic}.\n\n"
        "1. Gather relevant scientific and factual background information.\n"
        "2. Identify key concepts, mechanisms, and important data.\n"
        "3. If applicable, include recent developments or research findings.\n"
        "4. Provide structured research notes."
    ),
    expected_output=(
        "Structured research notes with the following sections:\n"
        "- Topic Overview\n"
        "- Key Concepts\n"
        "- Mechanisms / Causes\n"
        "- Recent Developments\n"
        "- Key Insights"
    ),
    agent=researcher,
    )

    write_task = Task(
    description=(
        "Using the Researcher's notes, write a comprehensive report on {topic}.\n"
        "The report should:\n"
        "- Have clear headings\n"
        "- Be logically structured\n"
        "- Explain concepts clearly\n"
        "- Avoid repetition\n"
        "- Be suitable for a non-expert audience"
    ),
    expected_output=(
        "A polished markdown report with:\n"
        "# Title\n"
        "## Introduction\n"
        "## Main Findings\n"
        "## Recent Developments\n"
        "## Conclusion"
    ),
    agent=writer,
    )

    review_task = Task(
    description=(
        "Review the report on {topic} for:\n"
        "- Factual accuracy\n"
        "- Logical consistency\n"
        "- Missing information\n"
        "- Clarity and readability\n\n"
        "Suggest specific improvements and provide a refined final version."
    ),
    expected_output=(
        "1. A bullet-point critique of weaknesses\n"
        "2. A revised and improved final report"
    ),
    agent=critic,
    )

    crew = Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, write_task, review_task],
        llm=llm,
        verbose=True,
    )
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(crew.kickoff, inputs={"topic": topic})
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            return "⚠️ Crew AI research took too long and was stopped."
        except Exception as e:
            return f"❌ An error occurred: {e}"

