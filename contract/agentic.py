# Improved Agentic RAG with LangGraph for Contracts and Addendums
import logging
import os
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
load_dotenv()

api_key = os.getenv("OPEN_AI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- State Schema ---
class ContractState(TypedDict):
    question: str
    partner: str
    main_answer: Optional[str]
    enriched_answer: Optional[str]
    final_answer: Optional[str]
    context_sources: List[Dict[str, Any]]
    confidence_score: float
    processing_steps: List[str]


# --- Configuration ---
class Config:
    MODEL_NAME = "gpt-4o"
    TEMPERATURE = 0
    MAX_VERSIONS = 20  # Configurable max versions
    MAIN_CONTRACT_VERSION = 1
    RETRIEVAL_K = 5
    ADDENDUM_K = 3
    MIN_CONFIDENCE_THRESHOLD = 0.7


# --- Setup LLM and Vectorstore ---
llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


# --- Helper Classes ---
class ContractRetriever:
    def __init__(self, vectorstore, embedding_function):
        self.vectorstore = vectorstore
        self.embedding = embedding_function

    def retrieve_with_metadata(self, query: str, partner: str, version: int, k: int = 5) -> List[Document]:
        """Retrieve documents with enhanced filtering and metadata"""
        try:
            docs = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter={"partner": partner, "version": version}
            )
            # Return documents with confidence scores
            return [(doc, score) for doc, score in docs if score > 0.3]  # Filter low-confidence results
        except Exception as e:
            logger.error(f"Retrieval error for partner {partner}, version {version}: {e}")
            return []

    def get_available_versions(self, partner: str) -> List[int]:
        """Get all available versions for a partner"""
        try:
            # This would need to be implemented based on your vectorstore's capabilities
            # For now, we'll check versions incrementally
            versions = []
            for v in range(1, Config.MAX_VERSIONS + 1):
                docs = self.vectorstore.similarity_search(
                    query="contract", k=1, filter={"partner": partner, "version": v}
                )
                if docs:
                    versions.append(v)
                else:
                    break  # Assume versions are sequential
            return versions
        except Exception as e:
            logger.error(f"Error getting versions for partner {partner}: {e}")
            return [1]  # Default to main contract only


retriever = ContractRetriever(vectorstore, embedding)


# --- Enhanced Agent Functions ---
def retrieve_main_contract(state: ContractState) -> ContractState:
    """Enhanced main contract retrieval with confidence scoring"""
    question = state["question"]
    partner = state["partner"]

    logger.info(f"Retrieving main contract for partner: {partner}")

    docs_with_scores = retriever.retrieve_with_metadata(
        query=question,
        partner=partner,
        version=Config.MAIN_CONTRACT_VERSION,
        k=Config.RETRIEVAL_K
    )

    if not docs_with_scores:
        logger.warning(f"No main contract found for partner: {partner}")
        return {
            **state,
            "main_answer": "No main contract found for this partner.",
            "context_sources": [],
            "confidence_score": 0.0,
            "processing_steps": state.get("processing_steps", []) + ["Main contract retrieval: No documents found"]
        }

    # Calculate confidence based on retrieval scores
    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
    confidence = min(1.0, (1.0 - avg_score))  # Convert distance to confidence

    content = "\n".join(doc.page_content for doc, _ in docs_with_scores)
    sources = [
        {
            "version": Config.MAIN_CONTRACT_VERSION,
            "score": float(score),
            "content_preview": doc.page_content[:100] + "..."
        }
        for doc, score in docs_with_scores
    ]

    # Enhanced prompt for main contract analysis
    system_prompt = """You are a contract analysis expert. Analyze the provided contract content and answer the user's question accurately and comprehensively. 
    If the information is insufficient to fully answer the question, clearly state what information is missing."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}\n\nContract Content:\n{content}")
    ]

    try:
        main_answer = llm.invoke(messages).content
    except Exception as e:
        logger.error(f"LLM error in main contract analysis: {e}")
        main_answer = "Error processing main contract content."
        confidence = 0.0

    return {
        **state,
        "main_answer": main_answer,
        "context_sources": sources,
        "confidence_score": confidence,
        "processing_steps": state.get("processing_steps", []) + [
            f"Main contract analyzed with confidence: {confidence:.2f}"]
    }


def retrieve_addendums(state: ContractState) -> ContractState:
    """Enhanced addendum processing with incremental enrichment"""
    question = state["question"]
    partner = state["partner"]
    main_answer = state.get("main_answer", "")

    if not main_answer or main_answer == "No main contract found for this partner.":
        # Try to find answer in addendums if main contract failed
        return search_all_versions(state)

    logger.info(f"Processing addendums for partner: {partner}")

    available_versions = retriever.get_available_versions(partner)
    addendum_versions = [v for v in available_versions if v > Config.MAIN_CONTRACT_VERSION]

    if not addendum_versions:
        logger.info(f"No addendums found for partner: {partner}")
        return {
            **state,
            "enriched_answer": main_answer,
            "processing_steps": state.get("processing_steps", []) + ["No addendums found"]
        }

    enriched_answer = main_answer
    all_sources = state.get("context_sources", [])
    processing_steps = state.get("processing_steps", [])

    for version in sorted(addendum_versions):
        docs_with_scores = retriever.retrieve_with_metadata(
            query=question,
            partner=partner,
            version=version,
            k=Config.ADDENDUM_K
        )

        if not docs_with_scores:
            continue

        context = "\n".join(doc.page_content for doc, _ in docs_with_scores)

        # Enhanced prompt for addendum analysis
        system_prompt = """You are a contract addendum analyst. Your task is to determine if the addendum content modifies, overrides, or supplements the current answer.

        Rules:
        1. If the addendum directly contradicts or overrides the current answer, provide the updated answer
        2. If the addendum supplements or adds new information, integrate it with the current answer
        3. If the addendum is not relevant to the question, return the original answer unchanged
        4. Always explain what changes (if any) were made"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            QUESTION: {question}
            CURRENT ANSWER: {enriched_answer}
            ADDENDUM CONTENT (Version {version}):\n{context}

            Provide the updated answer and explain any changes made.
            """)
        ]

        try:
            response = llm.invoke(messages).content
            enriched_answer = response

            # Add sources
            sources = [
                {
                    "version": version,
                    "score": float(score),
                    "content_preview": doc.page_content[:100] + "..."
                }
                for doc, score in docs_with_scores
            ]
            all_sources.extend(sources)
            processing_steps.append(f"Processed addendum version {version}")

        except Exception as e:
            logger.error(f"LLM error processing addendum version {version}: {e}")
            processing_steps.append(f"Error processing addendum version {version}")

    return {
        **state,
        "enriched_answer": enriched_answer,
        "context_sources": all_sources,
        "processing_steps": processing_steps
    }


def search_all_versions(state: ContractState) -> ContractState:
    """Fallback: Search through all versions if main contract doesn't have the answer"""
    question = state["question"]
    partner = state["partner"]

    logger.info(f"Searching all versions for partner: {partner}")

    available_versions = retriever.get_available_versions(partner)

    for version in sorted(available_versions):
        docs_with_scores = retriever.retrieve_with_metadata(
            query=question,
            partner=partner,
            version=version,
            k=Config.RETRIEVAL_K
        )

        if docs_with_scores:
            content = "\n".join(doc.page_content for doc, _ in docs_with_scores)

            system_prompt = """You are a contract analyst. Answer the user's question based on the provided contract content. 
            Be specific about which version/document the information comes from."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {question}\n\nContract Content (Version {version}):\n{content}")
            ]

            try:
                answer = llm.invoke(messages).content
                if "cannot" not in answer.lower() and "unable" not in answer.lower():
                    sources = [
                        {
                            "version": version,
                            "score": float(score),
                            "content_preview": doc.page_content[:100] + "..."
                        }
                        for doc, score in docs_with_scores
                    ]

                    return {
                        **state,
                        "enriched_answer": answer,
                        "context_sources": sources,
                        "processing_steps": state.get("processing_steps", []) + [f"Found answer in version {version}"]
                    }
            except Exception as e:
                logger.error(f"Error processing version {version}: {e}")

    return {
        **state,
        "enriched_answer": "No relevant information found in any contract version.",
        "context_sources": [],
        "processing_steps": state.get("processing_steps", []) + ["No relevant information found in any version"]
    }


def synthesize_final_answer(state: ContractState) -> ContractState:
    """Enhanced synthesis with quality assessment"""
    main_answer = state.get("main_answer", "")
    enriched_answer = state.get("enriched_answer", "")
    sources = state.get("context_sources", [])
    confidence = state.get("confidence_score", 0.0)

    # If we only have one answer or they're very similar, use the enriched version
    if not main_answer or main_answer == enriched_answer:
        final_answer = enriched_answer
    else:
        # Synthesize both answers
        system_prompt = """You are a contract synthesis expert. Your task is to create a comprehensive final answer by combining information from the main contract and addendums.

        Guidelines:
        1. Prioritize the most recent/relevant information
        2. Clearly indicate when information comes from addendums vs main contract
        3. Resolve any conflicts by favoring addendum information (as they typically override main contracts)
        4. Provide a clear, structured response"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            QUESTION: {state['question']}

            MAIN CONTRACT RESPONSE:
            {main_answer}

            ENRICHED/ADDENDUM RESPONSE:
            {enriched_answer}

            Please provide a final synthesized answer.
            """)
        ]

        try:
            final_answer = llm.invoke(messages).content
        except Exception as e:
            logger.error(f"Error in final synthesis: {e}")
            final_answer = enriched_answer  # Fallback to enriched answer

    # Calculate final confidence
    final_confidence = confidence
    if sources:
        avg_source_score = sum(s.get("score", 0) for s in sources) / len(sources)
        final_confidence = min(1.0, (confidence + (1.0 - avg_source_score)) / 2)

    return {
        **state,
        "final_answer": final_answer,
        "confidence_score": final_confidence,
        "processing_steps": state.get("processing_steps", []) + ["Final answer synthesized"]
    }


# --- Router Functions ---
def should_search_all_versions(state: ContractState) -> str:
    """Route to comprehensive search if main contract is insufficient"""
    main_answer = state.get("main_answer", "")
    confidence = state.get("confidence_score", 0.0)

    if (not main_answer or
            "No main contract found" in main_answer or
            confidence < Config.MIN_CONFIDENCE_THRESHOLD):
        return "SearchAllVersions"
    return "AddendumAgent"


# --- Define LangGraph ---
workflow = StateGraph(ContractState)

# Add nodes
workflow.add_node("MainContractAgent", retrieve_main_contract)
workflow.add_node("AddendumAgent", retrieve_addendums)
workflow.add_node("SearchAllVersions", search_all_versions)
workflow.add_node("FinalSynthesizer", synthesize_final_answer)

# Define workflow
workflow.set_entry_point("MainContractAgent")
workflow.add_conditional_edges(
    "MainContractAgent",
    should_search_all_versions,
    {
        "AddendumAgent": "AddendumAgent",
        "SearchAllVersions": "SearchAllVersions"
    }
)
workflow.add_edge("AddendumAgent", "FinalSynthesizer")
workflow.add_edge("SearchAllVersions", "FinalSynthesizer")
workflow.add_edge("FinalSynthesizer", END)

graph = workflow.compile()


# --- Enhanced Execution Function ---
def process_contract_query(question: str, partner: str, verbose: bool = True) -> Dict[str, Any]:
    """Process a contract query with enhanced error handling and logging"""
    start_time = datetime.now()

    try:
        result = graph.invoke(
            {
                "question": question,
                "partner": partner,
                "context_sources": [],
                "confidence_score": 0.0,
                "processing_steps": []
            },
            config=RunnableConfig()
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"QUESTION: {question}")
            print(f"PARTNER: {partner}")
            print(f"{'=' * 50}")
            print(f"\nFINAL ANSWER:")
            print(result["final_answer"])
            print(f"\nCONFIDENCE SCORE: {result.get('confidence_score', 0.0):.2f}")
            print(f"PROCESSING TIME: {processing_time:.2f}s")
            print(f"\nSOURCES USED: {len(result.get('context_sources', []))}")
            for i, source in enumerate(result.get('context_sources', [])[:3]):  # Show first 3 sources
                print(f"  {i + 1}. Version {source['version']} (Score: {source['score']:.3f})")
            print(f"\nPROCESSING STEPS:")
            for step in result.get('processing_steps', []):
                print(f"  - {step}")

        return result

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "question": question,
            "partner": partner,
            "final_answer": f"Error processing query: {str(e)}",
            "confidence_score": 0.0,
            "context_sources": [],
            "processing_steps": [f"Error: {str(e)}"]
        }


# --- Example Usage ---
if __name__ == "__main__":
    # Test cases
    test_queries = [
        ("What are the Expenses for codecraft?", "codecraft"),
        # ("What is the payment schedule?", "PartnerY"),
        # ("Are there any liability limitations?", "PartnerZ")
    ]

    for question, partner in test_queries:
        result = process_contract_query(question, partner, verbose=True)
        print("\n" + "=" * 80 + "\n")