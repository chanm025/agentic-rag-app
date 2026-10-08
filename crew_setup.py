import concurrent.futures

from crewai import Agent, Crew, Process, Task

from llm_adapter import llm
from tools import rag_pdf_search, tavily_tool


def run_crew(topic: str, timeout=180):
    """Run researcher -> writer -> reviewer on a topic and return the final report as text."""

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
        tools=[rag_pdf_search, tavily_tool],   # the researcher is the only agent that retrieves
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal=(
            "Generate clear, structured, well-organised answers or reports based "
            "on the Researcher's findings."
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
            "Review the Writer's answer for factual accuracy, coherence, and "
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
            "1. Search the PDF knowledge base and the web for relevant scientific and factual background.\n"
            "2. Identify key concepts, mechanisms, and important data.\n"
            "3. If applicable, include recent developments or research findings.\n"
            "4. Provide structured research notes and say which source each finding came from."
        ),
        expected_output=(
            "Structured research notes with the following sections:\n"
            "- Topic Overview\n- Key Concepts\n- Mechanisms / Causes\n"
            "- Recent Developments\n- Key Insights"
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the Researcher's notes, write a comprehensive report on {topic}.\n"
            "The report should:\n"
            "- Have clear headings\n- Be logically structured\n- Explain concepts clearly\n"
            "- Avoid repetition\n- Be suitable for a non-expert audience"
        ),
        expected_output=(
            "A polished markdown report with:\n"
            "# Title\n## Introduction\n## Main Findings\n## Recent Developments\n## Conclusion"
        ),
        agent=writer,
    )

    review_task = Task(
        description=(
            "Review the report on {topic} for:\n"
            "- Factual accuracy\n- Logical consistency\n- Missing information\n"
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
        process=Process.sequential,   # each task receives the previous task's output
        verbose=True,
    )

    # Run in a worker thread so the UI can give up after `timeout` seconds. A thread cannot be
    # killed, so a timed-out run keeps going in the background; we just stop waiting for it.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(crew.kickoff, inputs={"topic": topic})
    try:
        return str(future.result(timeout=timeout))
    except concurrent.futures.TimeoutError:
        return "⚠️ The research took too long and was stopped waiting. Try a narrower topic."
    except Exception as e:
        return f"❌ An error occurred: {e}"
    finally:
        executor.shutdown(wait=False)
