import os

import streamlit as st

st.set_page_config(page_title="AI Research Assistant", layout="wide")  # must be the first Streamlit call

from crew_setup import run_crew  # noqa: E402  (imports llm_adapter, which loads the .env file)

st.title("🔬 AI Research Assistant")
st.caption("A three-agent crew (researcher → writer → reviewer) that searches a local PDF knowledge base and the web.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set: add it to a .env file (see .env.example).")
if not os.getenv("TAVILY_API_KEY"):
    st.info("TAVILY_API_KEY is not set: web search will be skipped and only the PDFs will be used.")

topic = st.text_input("Enter research topic")

if st.button("Run Research") and topic:
    with st.spinner("Running multi-agent research (this can take a minute or two)..."):
        result = run_crew(topic, timeout=180)

    st.markdown("## 📄 Final Report")
    st.markdown(result)
