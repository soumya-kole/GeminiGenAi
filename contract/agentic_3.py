import os
from typing import List, Optional, Dict, Any, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor


# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")

# ---------------------
# Configuration
# ---------------------
class Config:
    MODEL = "gpt-4o"
    TEMPERATURE = 0
    RETRIEVAL_K = 5
    ADDENDUM_K = 3

# ---------------------
# LangChain setup
# ---------------------
llm = ChatOpenAI(model=Config.MODEL, temperature=Config.TEMPERATURE)
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# ---------------------
# State definition
# ---------------------
class ContractState(TypedDict):
    question: str
    partner: str
    main_answer: Optional[str]
    enriched_answer: Optional[str]
    final_answer: Optional[str]
    context_sources: List[Dict[str, Any]]
    confidence_score: float
    processing_steps: List[str]

# ---------------------
# Tool Definitions
# ---------------------
@tool
def retrieve_main_contract_docs(question: str, partner: str) -> List[str]:
    """Retrieve relevant content from the main contract (version 1) for a given question and partner."""
    partner = partner.lower()
    filter_dict = {"$and": [{"partner": partner}, {"version": 1}]}
    docs_with_scores = vectorstore.similarity_search_with_score(
        question, k=Config.RETRIEVAL_K, filter=filter_dict
    )
    return [doc.page_content for doc, score in docs_with_scores if score > 0.3]

@tool
def retrieve_addendum_docs(question: str, partner: str, version: int) -> List[str]:
    """Retrieve content from a specific addendum version for a given question and partner."""
    partner = partner.lower()
    filter_dict = {"$and": [{"partner": partner}, {"version": version}]}
    docs_with_scores = vectorstore.similarity_search_with_score(
        question, k=Config.ADDENDUM_K, filter=filter_dict
    )
    return [doc.page_content for doc, score in docs_with_scores if score > 0.3]

# ---------------------
# Agent Definitions
# ---------------------
def build_main_contract_agent():
    tools = [retrieve_main_contract_docs]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful legal agent. Use the provided tools to retrieve and summarize the main contract."),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def build_addendum_agent():
    tools = [retrieve_addendum_docs]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an addendum analyst. Use tools to fetch addendum documents and determine whether they override or enhance the main contract."),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------
# LangGraph Node Functions
# ---------------------
main_agent = build_main_contract_agent()
addendum_agent = build_addendum_agent()

def main_contract_node(state: ContractState) -> ContractState:
    response = main_agent.invoke({
        "input": f"Get the answer to: {state['question']}. Partner: {state['partner']}"
    })
    return {
        **state,
        "main_answer": response.get("output"),
        "processing_steps": state.get("processing_steps", []) + ["MainContractAgent executed"]
    }

def addendum_node(state: ContractState) -> ContractState:
    response = addendum_agent.invoke({
        "input": f"Check if any addendum updates this answer: {state['main_answer']}"
    })
    return {
        **state,
        "enriched_answer": response.get("output"),
        "processing_steps": state.get("processing_steps", []) + ["AddendumAgent executed"]
    }

def final_synth_node(state: ContractState) -> ContractState:
    prompt = f"""
    Synthesize final answer for question: {state['question']}
    Main: {state.get('main_answer')}
    Addendum: {state.get('enriched_answer')}
    """
    final = llm.invoke([{"role": "user", "content": prompt}]).content
    return {
        **state,
        "final_answer": final,
        "processing_steps": state.get("processing_steps", []) + ["FinalSynthesizer executed"]
    }

# ---------------------
# LangGraph Workflow
# ---------------------
graph = StateGraph(ContractState)
graph.set_entry_point("MainContractAgent")

graph.add_node("MainContractAgent", main_contract_node)
graph.add_node("AddendumAgent", addendum_node)
graph.add_node("FinalSynthesizer", final_synth_node)

graph.add_edge("MainContractAgent", "AddendumAgent")
graph.add_edge("AddendumAgent", "FinalSynthesizer")
graph.add_edge("FinalSynthesizer", END)

workflow = graph.compile()

# ---------------------
# Execution Function
# ---------------------
def run_contract_query(question: str, partner: str):
    return workflow.invoke({
        "question": question,
        "partner": partner,
        "context_sources": [],
        "confidence_score": 0.0,
        "processing_steps": []
    })

# ---------------------
# Example Usage
# ---------------------
if __name__ == "__main__":
    result = run_contract_query("What is the Total Project Fee?", "codecraft")
    print("Final Answer:", result["final_answer"])
    print("Steps:")
    for step in result["processing_steps"]:
        print(" -", step)
