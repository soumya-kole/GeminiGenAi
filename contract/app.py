import os
from typing import List, Optional, Dict, Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from agentic_4 import workflow
# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")


# Configuration
class Config:
    MODEL = "gpt-4o"
    TEMPERATURE = 0
    RETRIEVAL_K = 5
    ADDENDUM_K = 3


# LangChain setup
llm = ChatOpenAI(model=Config.MODEL, temperature=Config.TEMPERATURE)
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


# State definition
class ContractState(TypedDict):
    question: str
    partner: str
    main_answer: Optional[str]
    main_sources: List[Dict[str, Any]]
    addendum_results: List[Dict[str, Any]]
    enriched_answer: Optional[str]
    final_answer: Optional[str]
    context_sources: List[Dict[str, Any]]
    confidence_score: float
    processing_steps: List[str]
    override_detected: bool


# [Include all the tool definitions, agent builders, node functions, and workflow setup from the enhanced version]

# The main function you need:
def run_contract_query(question: str, partner: str):
    """Run a contract query with enhanced addendum override detection."""
    initial_state = {
        "question": question,
        "partner": partner,
        "main_sources": [],
        "addendum_results": [],
        "context_sources": [],
        "confidence_score": 0.0,
        "processing_steps": [],
        "override_detected": False
    }

    result = workflow.invoke(initial_state)
    return result


# Set page config
st.set_page_config(
    page_title="Contract Query Assistant",
    page_icon="📋",
    layout="centered"
)

# Title
st.title("📋 Contract Query Assistant")
st.write("Ask questions about your contracts and get answers from both main contracts and addendums.")

# Input form
with st.form("query_form"):
    st.subheader("Ask Your Question")

    # Partner selection
    partner = st.text_input(
        "Partner Name:",
        placeholder="e.g., codecraft",
        help="Enter the partner name (case insensitive)"
    )

    # Question input
    question = st.text_area(
        "Question:",
        placeholder="e.g., What is the Total Project Fee?",
        help="Enter your question about the contract"
    )

    # Submit button
    submitted = st.form_submit_button("Get Answer", type="primary")

# Process query when form is submitted
if submitted:
    if not partner.strip():
        st.error("Please enter a partner name.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Initialize
            progress_bar.progress(10)
            status_text.text("🔍 Initializing contract analysis...")

            # Step 2: Main Contract Analysis
            progress_bar.progress(30)
            status_text.text("📋 Analyzing main contract...")

            # Create a placeholder for intermediate updates
            result_placeholder = st.empty()

            # Run the query (this is where the actual work happens)
            result = run_contract_query(question.strip(), partner.strip())

            # Step 3: Addendum Analysis
            progress_bar.progress(70)
            status_text.text("📑 Checking addendums for overrides...")

            # Step 4: Final Synthesis
            progress_bar.progress(90)
            status_text.text("🔄 Synthesizing final answer...")

            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            # Small delay to show completion
            import time

            time.sleep(0.5)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Display results
            st.success("Analysis Complete!")

            # Final Answer
            st.subheader("📝 Answer")
            st.write(result.get("final_answer", "No answer found."))

            # Override status
            if result.get("override_detected", False):
                st.info(
                    "⚠️ **Override Detected**: This answer comes from an addendum that overrides the main contract.")
            else:
                st.info("ℹ️ **Main Contract**: This answer is based on the main contract provisions.")

            # Optional: Show processing steps in an expander
            with st.expander("View Processing Details"):
                st.write("**Processing Steps:**")
                for i, step in enumerate(result.get("processing_steps", []), 1):
                    st.write(f"{i}. {step}")

                if result.get("main_answer"):
                    st.write("**Main Contract Analysis:**")
                    st.write(result["main_answer"])

                if result.get("enriched_answer"):
                    st.write("**Addendum Analysis:**")
                    st.write(result["enriched_answer"])

        except Exception as e:
            # Clear progress indicators on error
            progress_bar.empty()
            status_text.empty()
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your configuration and try again.")

# Footer
st.markdown("---")
st.markdown("*Ask questions about contracts, fees, terms, conditions, or any other contract provisions.*")