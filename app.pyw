import streamlit as st
from crew_setup import run_crew

st.title("🔬 AI Research Assistant")

st.set_page_config(page_title="AI Research Assistant", layout="wide")

topic = st.text_input("Enter research topic")

if st.button("Run Research") and topic:
    with st.spinner("Running multi-agent research..."):
        result = run_crew(topic, timeout=60)

    st.markdown("## 📄 Final Report")
    st.markdown(result)
