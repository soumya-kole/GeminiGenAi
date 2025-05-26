# Improved Agentic RAG with LangGraph for Contracts and Addendums
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

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
    MIN_CONFIDENCE_THRESHOLD = 0.3


# --- Setup LLM and Vectorstore ---
llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)


# --- Helper Classes ---
class ContractRetriever:
    def __init__(self, vectorstore, embedding_function):
        self.vectorstore = vectorstore
        self.embedding = embedding_function
        self._test_filter_format()

    def _test_filter_format(self):
        """Test which filter format works with this Chroma instance"""
        self.filter_format = "dict"  # Default
        try:
            # Test simple dict format first
            test_docs = self.vectorstore.similarity_search(
                query="test", k=1, filter={"partner": "test"}
            )
            logger.info("Using dict filter format")
        except Exception as e1:
            try:
                # Try individual key-value format
                test_docs = self.vectorstore.similarity_search(
                    query="test", k=1,
                    filter={"$and": [{"partner": {"$eq": "test"}}]}
                )
                self.filter_format = "mongodb"
                logger.info("Using MongoDB-style filter format")
            except Exception as e2:
                try:
                    # Try where clause format
                    test_docs = self.vectorstore.similarity_search(
                        query="test", k=1,
                        where={"partner": "test"}
                    )
                    self.filter_format = "where"
                    logger.info("Using where clause format")
                except Exception as e3:
                    # Fallback to no filtering
                    self.filter_format = "none"
                    logger.warning("No filtering support detected, will filter post-retrieval")

    def _create_filter(self, partner: str, version: int):
        """Create appropriate filter based on detected format"""
        if self.filter_format == "dict":
            return {"partner": partner, "version": version}
        elif self.filter_format == "mongodb":
            return {
                "$and": [
                    {"partner": {"$eq": partner}},
                    {"version": {"$eq": version}}
                ]
            }
        elif self.filter_format == "where":
            return {"partner": partner, "version": version}
        else:
            return None

    def retrieve_with_metadata(self, query: str, partner: str, version: int, k: int = 5) -> List[Document]:
        """Retrieve documents with enhanced filtering and metadata"""
        try:
            filter_dict = self._create_filter(partner, version)

            if self.filter_format == "where":
                # Use where parameter instead of filter
                docs = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k * 3,  # Get more docs to filter manually if needed
                    where=filter_dict
                )
            elif filter_dict:
                # Use filter parameter
                docs = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k * 3,  # Get more docs to filter manually if needed
                    filter=filter_dict
                )
            else:
                # No filtering support - retrieve more and filter manually
                docs = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k * 10  # Get many more for manual filtering
                )
                # Manual filtering
                filtered_docs = []
                for doc, score in docs:
                    doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    if (doc_metadata.get('partner') == partner and
                            doc_metadata.get('version') == version):
                        filtered_docs.append((doc, score))
                docs = filtered_docs[:k]

            # Return documents with confidence scores (filter low-confidence results)
            return [(doc, score) for doc, score in docs[:k] if score > 0.3]

        except Exception as e:
            logger.error(f"Retrieval error for partner {partner}, version {version}: {e}")
            # Fallback: try without filtering
            try:
                docs = self.vectorstore.similarity_search_with_score(query=query, k=k * 5)
                # Manual filtering
                filtered_docs = []
                for doc, score in docs:
                    doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    if (doc_metadata.get('partner') == partner and
                            doc_metadata.get('version') == version):
                        filtered_docs.append((doc, score))
                ret =  [(doc, score) for doc, score in filtered_docs[:k]
                        # if score > 0.3
                        ]
                logger.info(f"Fallback retrieval successful, found {len(ret)} documents")
                return ret
            except Exception as e2:
                logger.error(f"Fallback retrieval also failed: {e2}")
                return []

    def get_available_versions(self, partner: str) -> List[int]:
        """Get all available versions for a partner"""
        try:
            versions = set()

            # Try to get a large sample of documents for this partner
            if self.filter_format == "where":
                docs = self.vectorstore.similarity_search(
                    query="contract document", k=100,
                    where={"partner": partner}
                )
            elif self.filter_format != "none":
                filter_dict = {"partner": partner} if self.filter_format == "dict" else {
                    "$and": [{"partner": {"$eq": partner}}]
                }
                docs = self.vectorstore.similarity_search(
                    query="contract document", k=100,
                    filter=filter_dict
                )
            else:
                # No filtering - get many docs and filter manually
                docs = self.vectorstore.similarity_search(query="contract document", k=500)
                docs = [doc for doc in docs
                        if hasattr(doc, 'metadata') and
                        doc.metadata.get('partner') == partner]

            # Extract versions from metadata
            for doc in docs:
                if hasattr(doc, 'metadata') and 'version' in doc.metadata:
                    versions.add(doc.metadata['version'])

            available_versions = sorted(list(versions))

            if not available_versions:
                # Fallback: try sequential checking
                logger.info(f"No versions found via metadata search, trying sequential check for {partner}")
                for v in range(1, min(Config.MAX_VERSIONS + 1, 10)):  # Limit to 10 for performance
                    test_docs = self.retrieve_with_metadata("contract", partner, v, k=1)
                    if test_docs:
                        available_versions.append(v)
                    elif v > 3:  # If we don't find anything after version 3, likely no more
                        break

            return available_versions if available_versions else [1]  # Default to main contract

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
            for i, source in enumerate(result.get('context_sources', [])[:10]):  # Show first 10 sources
                print(f"  {i + 1}. Version {source['version']} (Score: {source['score']:.3f})")
                print(result)
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
        ("What is the Total Project Fee for codecraft?", "codecraft"),
        # ("What is the payment schedule?", "PartnerY"),
        # ("Are there any liability limitations?", "PartnerZ")
    ]

    for question, partner in test_queries:
        result = process_contract_query(question, partner, verbose=True)
        print("\n" + "=" * 80 + "\n")