import os
from typing import List, Optional, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

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
    main_sources: List[Dict[str, Any]]
    addendum_results: List[Dict[str, Any]]
    enriched_answer: Optional[str]
    final_answer: Optional[str]
    context_sources: List[Dict[str, Any]]
    confidence_score: float
    processing_steps: List[str]
    override_detected: bool


# ---------------------
# Enhanced Tool Definitions
# ---------------------
@tool
def retrieve_main_contract_docs(question: str, partner: str) -> Dict[str, Any]:
    """Retrieve relevant content from the main contract (version 1) for a given question and partner."""
    partner = partner.lower()
    filter_dict = {"$and": [{"partner": partner}, {"version": 1}]}
    docs_with_scores = vectorstore.similarity_search_with_score(
        question, k=Config.RETRIEVAL_K, filter=filter_dict
    )

    relevant_docs = []
    for doc, score in docs_with_scores:
        if score < 0.7:  # Lower threshold means higher similarity
            relevant_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
                "version": doc.metadata.get("version", 1)
            })

    return {
        "documents": relevant_docs,
        "count": len(relevant_docs),
        "partner": partner,
        "version": 1
    }


@tool
def retrieve_all_addendum_versions(question: str, partner: str) -> Dict[str, Any]:
    """Retrieve content from ALL addendum versions for a given question and partner to detect overrides."""
    partner = partner.lower()

    # Get all versions for this partner (excluding version 1 which is main contract)
    all_versions_filter = {"partner": partner}
    all_docs = vectorstore.similarity_search_with_score(
        question, k=50, filter=all_versions_filter  # Get more docs to ensure we catch all versions
    )

    # Group by version and filter by relevance
    version_docs = {}
    for doc, score in all_docs:
        version = doc.metadata.get("version", 1)
        if version > 1 and score < 0.7:  # Only addendums (version > 1) with good similarity
            if version not in version_docs:
                version_docs[version] = []
            version_docs[version].append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
                "version": version
            })

    return {
        "addendum_versions": version_docs,
        "versions_found": list(version_docs.keys()),
        "partner": partner,
        "total_versions": len(version_docs)
    }


@tool
def detect_override_keywords(text: str) -> Dict[str, Any]:
    """Detect if text contains override/amendment keywords."""
    override_keywords = [
        "override", "overrides", "replaces", "supersedes", "amends", "amended",
        "modified", "changes", "updates", "revised", "new provision",
        "instead of", "rather than", "in lieu of", "substitute"
    ]

    text_lower = text.lower()
    found_keywords = [keyword for keyword in override_keywords if keyword in text_lower]

    return {
        "has_override_language": len(found_keywords) > 0,
        "override_keywords": found_keywords,
        "override_score": len(found_keywords)
    }


# ---------------------
# Enhanced Agent Definitions
# ---------------------
def build_main_contract_agent():
    tools = [retrieve_main_contract_docs]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal contract analyzer focused on main contract provisions.

        Your task is to:
        1. Retrieve relevant sections from the main contract (version 1)
        2. Provide a clear, factual answer based on the main contract
        3. Extract specific values, terms, and conditions mentioned
        4. Note any ambiguities or areas that might need clarification

        Be precise and quote relevant sections when possible."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def build_addendum_agent():
    tools = [retrieve_all_addendum_versions, detect_override_keywords]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an addendum analysis specialist. Your critical task is to detect when addendums override or modify main contract provisions.

        For each question, you must:
        1. Search ALL addendum versions for relevant content
        2. Analyze each addendum for override language (replaces, supersedes, amends, etc.)
        3. Compare addendum provisions with the main contract answer
        4. Determine if any addendum OVERRIDES the main contract
        5. If multiple addendums exist, identify which is the most recent/authoritative

        IMPORTANT: Look for explicit override language like:
        - "This replaces section X"
        - "Supersedes the original provision"
        - "Amends the contract to change X to Y"
        - "Instead of X, the new provision is Y"

        Always check the version numbers and dates to determine precedence."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ---------------------
# Enhanced LangGraph Node Functions
# ---------------------
main_agent = build_main_contract_agent()
addendum_agent = build_addendum_agent()


def main_contract_node(state: ContractState) -> ContractState:
    response = main_agent.invoke({
        "input": f"Analyze the main contract to answer: '{state['question']}' for partner '{state['partner']}'. Provide specific details and quote relevant sections."
    })

    return {
        **state,
        "main_answer": response.get("output"),
        "processing_steps": state.get("processing_steps", []) + [
            "MainContractAgent executed - analyzed main contract provisions"]
    }


def addendum_node(state: ContractState) -> ContractState:
    # Enhanced input with more context
    enhanced_input = f"""
    MAIN CONTRACT ANSWER: {state.get('main_answer', 'No main answer found')}

    QUESTION: {state['question']}
    PARTNER: {state['partner']}

    TASK: Search all addendum versions for this partner and determine if any addendum OVERRIDES or MODIFIES the main contract answer above. 

    Pay special attention to:
    1. Direct contradictions to the main contract answer
    2. Override language (replaces, supersedes, amends, etc.)
    3. New values that differ from the main contract
    4. Version numbers and effective dates

    If you find an override, clearly state what is being overridden and what the new provision is.
    """

    response = addendum_agent.invoke({"input": enhanced_input})

    # Check if override was detected in the response
    response_text = response.get("output", "").lower()
    override_indicators = ["override", "replaces", "supersedes", "amends", "modified", "changes", "instead of"]
    override_detected = any(indicator in response_text for indicator in override_indicators)

    return {
        **state,
        "enriched_answer": response.get("output"),
        "override_detected": override_detected,
        "processing_steps": state.get("processing_steps", []) + [
            f"AddendumAgent executed - {'Override detected' if override_detected else 'No override found'}"
        ]
    }


def final_synth_node(state: ContractState) -> ContractState:
    # Enhanced synthesis prompt
    synthesis_prompt = f"""
    QUESTION: {state['question']}
    PARTNER: {state['partner']}

    MAIN CONTRACT ANSWER:
    {state.get('main_answer', 'No main contract answer available')}

    ADDENDUM ANALYSIS:
    {state.get('enriched_answer', 'No addendum analysis available')}

    OVERRIDE DETECTED: {state.get('override_detected', False)}

    INSTRUCTIONS:
    Provide a final, authoritative answer that:
    1. If an override was detected, prioritize the addendum provision over the main contract
    2. If no override, use the main contract answer but note any addendum clarifications
    3. Be explicit about which document (main contract vs addendum) provides the final answer
    4. Include version information when relevant
    5. Flag any conflicts or ambiguities

    Format your response clearly with the final answer first, followed by the source and reasoning.
    """

    final_response = llm.invoke([{"role": "user", "content": synthesis_prompt}])

    return {
        **state,
        "final_answer": final_response.content,
        "processing_steps": state.get("processing_steps", []) + [
            f"FinalSynthesizer executed - {'Applied addendum override' if state.get('override_detected') else 'Used main contract answer'}"
        ]
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
# Enhanced Execution Function
# ---------------------
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


# ---------------------
# Debug Function
# ---------------------
def debug_contract_query(question: str, partner: str):
    """Run contract query with detailed debugging information."""
    print(f"\n=== DEBUG: Contract Query ===")
    print(f"Question: {question}")
    print(f"Partner: {partner}")
    print("=" * 50)

    result = run_contract_query(question, partner)

    print(f"\nProcessing Steps:")
    for i, step in enumerate(result.get("processing_steps", []), 1):
        print(f"  {i}. {step}")

    print(f"\nOverride Detected: {result.get('override_detected', False)}")
    print(f"\nMain Contract Answer:")
    print(f"  {result.get('main_answer', 'N/A')}")

    print(f"\nAddendum Analysis:")
    print(f"  {result.get('enriched_answer', 'N/A')}")

    print(f"\nFinal Answer:")
    print(f"  {result.get('final_answer', 'N/A')}")

    return result



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
    main_sources: List[Dict[str, Any]]
    addendum_results: List[Dict[str, Any]]
    enriched_answer: Optional[str]
    final_answer: Optional[str]
    context_sources: List[Dict[str, Any]]
    confidence_score: float
    processing_steps: List[str]
    override_detected: bool


# ---------------------
# Enhanced Tool Definitions
# ---------------------
@tool
def retrieve_main_contract_docs(question: str, partner: str) -> Dict[str, Any]:
    """Retrieve relevant content from the main contract (version 1) for a given question and partner."""
    partner = partner.lower()
    filter_dict = {"$and": [{"partner": partner}, {"version": 1}]}
    docs_with_scores = vectorstore.similarity_search_with_score(
        question, k=Config.RETRIEVAL_K, filter=filter_dict
    )

    relevant_docs = []
    for doc, score in docs_with_scores:
        if score < 0.7:  # Lower threshold means higher similarity
            relevant_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
                "version": doc.metadata.get("version", 1)
            })

    return {
        "documents": relevant_docs,
        "count": len(relevant_docs),
        "partner": partner,
        "version": 1
    }


@tool
def retrieve_all_addendum_versions(question: str, partner: str) -> Dict[str, Any]:
    """Retrieve content from ALL addendum versions for a given question and partner to detect overrides."""
    partner = partner.lower()

    # Get all versions for this partner (excluding version 1 which is main contract)
    all_versions_filter = {"partner": partner}
    all_docs = vectorstore.similarity_search_with_score(
        question, k=50, filter=all_versions_filter  # Get more docs to ensure we catch all versions
    )

    # Group by version and filter by relevance
    version_docs = {}
    for doc, score in all_docs:
        version = doc.metadata.get("version", 1)
        if version > 1 and score < 0.7:  # Only addendums (version > 1) with good similarity
            if version not in version_docs:
                version_docs[version] = []
            version_docs[version].append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
                "version": version
            })

    return {
        "addendum_versions": version_docs,
        "versions_found": list(version_docs.keys()),
        "partner": partner,
        "total_versions": len(version_docs)
    }


@tool
def detect_override_keywords(text: str) -> Dict[str, Any]:
    """Detect if text contains override/amendment keywords."""
    override_keywords = [
        "override", "overrides", "replaces", "supersedes", "amends", "amended",
        "modified", "changes", "updates", "revised", "new provision",
        "instead of", "rather than", "in lieu of", "substitute"
    ]

    text_lower = text.lower()
    found_keywords = [keyword for keyword in override_keywords if keyword in text_lower]

    return {
        "has_override_language": len(found_keywords) > 0,
        "override_keywords": found_keywords,
        "override_score": len(found_keywords)
    }


# ---------------------
# Enhanced Agent Definitions
# ---------------------
def build_main_contract_agent():
    tools = [retrieve_main_contract_docs]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a legal contract analyzer focused on main contract provisions.

        Your task is to:
        1. Retrieve relevant sections from the main contract (version 1)
        2. Provide a clear, factual answer based on the main contract
        3. Extract specific values, terms, and conditions mentioned
        4. Note any ambiguities or areas that might need clarification

        Be precise and quote relevant sections when possible."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def build_addendum_agent():
    tools = [retrieve_all_addendum_versions, detect_override_keywords]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an addendum analysis specialist. Your critical task is to detect when addendums override or modify main contract provisions.

        For each question, you must:
        1. Search ALL addendum versions for relevant content
        2. Analyze each addendum for override language (replaces, supersedes, amends, etc.)
        3. Compare addendum provisions with the main contract answer
        4. Determine if any addendum OVERRIDES the main contract
        5. If multiple addendums exist, identify which is the most recent/authoritative

        IMPORTANT: Look for explicit override language like:
        - "This replaces section X"
        - "Supersedes the original provision"
        - "Amends the contract to change X to Y"
        - "Instead of X, the new provision is Y"

        Always check the version numbers and dates to determine precedence."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ---------------------
# Enhanced LangGraph Node Functions
# ---------------------
main_agent = build_main_contract_agent()
addendum_agent = build_addendum_agent()


def main_contract_node(state: ContractState) -> ContractState:
    response = main_agent.invoke({
        "input": f"Analyze the main contract to answer: '{state['question']}' for partner '{state['partner']}'. Provide specific details and quote relevant sections."
    })

    return {
        **state,
        "main_answer": response.get("output"),
        "processing_steps": state.get("processing_steps", []) + [
            "MainContractAgent executed - analyzed main contract provisions"]
    }


def addendum_node(state: ContractState) -> ContractState:
    # Enhanced input with more context
    enhanced_input = f"""
    MAIN CONTRACT ANSWER: {state.get('main_answer', 'No main answer found')}

    QUESTION: {state['question']}
    PARTNER: {state['partner']}

    TASK: Search all addendum versions for this partner and determine if any addendum OVERRIDES or MODIFIES the main contract answer above. 

    Pay special attention to:
    1. Direct contradictions to the main contract answer
    2. Override language (replaces, supersedes, amends, etc.)
    3. New values that differ from the main contract
    4. Version numbers and effective dates

    If you find an override, clearly state what is being overridden and what the new provision is.
    """

    response = addendum_agent.invoke({"input": enhanced_input})

    # Check if override was detected in the response
    response_text = response.get("output", "").lower()
    override_indicators = ["override", "replaces", "supersedes", "amends", "modified", "changes", "instead of"]
    override_detected = any(indicator in response_text for indicator in override_indicators)

    return {
        **state,
        "enriched_answer": response.get("output"),
        "override_detected": override_detected,
        "processing_steps": state.get("processing_steps", []) + [
            f"AddendumAgent executed - {'Override detected' if override_detected else 'No override found'}"
        ]
    }


def final_synth_node(state: ContractState) -> ContractState:
    # Enhanced synthesis prompt
    synthesis_prompt = f"""
    QUESTION: {state['question']}
    PARTNER: {state['partner']}

    MAIN CONTRACT ANSWER:
    {state.get('main_answer', 'No main contract answer available')}

    ADDENDUM ANALYSIS:
    {state.get('enriched_answer', 'No addendum analysis available')}

    OVERRIDE DETECTED: {state.get('override_detected', False)}

    INSTRUCTIONS:
    Provide a final, authoritative answer that:
    1. If an override was detected, prioritize the addendum provision over the main contract
    2. If no override, use the main contract answer but note any addendum clarifications
    3. Be explicit about which document (main contract vs addendum) provides the final answer
    4. Include version information when relevant
    5. Flag any conflicts or ambiguities

    Format your response clearly with the final answer first, followed by the source and reasoning.
    """

    final_response = llm.invoke([{"role": "user", "content": synthesis_prompt}])

    return {
        **state,
        "final_answer": final_response.content,
        "processing_steps": state.get("processing_steps", []) + [
            f"FinalSynthesizer executed - {'Applied addendum override' if state.get('override_detected') else 'Used main contract answer'}"
        ]
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
print(workflow.get_graph().draw_mermaid())

# ---------------------
# Enhanced Execution Function
# ---------------------
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


# ---------------------
# Debug Function
# ---------------------
def debug_contract_query(question: str, partner: str):
    """Run contract query with detailed debugging information."""
    print(f"\n=== DEBUG: Contract Query ===")
    print(f"Question: {question}")
    print(f"Partner: {partner}")
    print("=" * 50)

    result = run_contract_query(question, partner)

    print(f"\nProcessing Steps:")
    for i, step in enumerate(result.get("processing_steps", []), 1):
        print(f"  {i}. {step}")

    print(f"\nOverride Detected: {result.get('override_detected', False)}")
    print(f"\nMain Contract Answer:")
    print(f"  {result.get('main_answer', 'N/A')}")

    print(f"\nAddendum Analysis:")
    print(f"  {result.get('enriched_answer', 'N/A')}")

    print(f"\nFinal Answer:")
    print(f"  {result.get('final_answer', 'N/A')}")

    return result


# ---------------------
# Example Usage
# ---------------------
if __name__ == "__main__":
    # Regular usage
    result = run_contract_query("What is the Total Project Fee?", "codecraft")
    print("Final Answer:", result["final_answer"])
    print("Override Detected:", result.get("override_detected", False))

    # Debug usage
    # debug_result = debug_contract_query("What is the Total Project Fee?", "codecraft")