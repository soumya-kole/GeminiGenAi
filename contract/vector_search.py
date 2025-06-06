import os
import warnings
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.getenv("OPEN_AI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
def search_relevant_chunks(question, partner, k=5, persist_directory="./chroma_db"):
    embedding_model = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    # Search with partner filter
    results = vectordb.similarity_search_with_relevance_scores(
        query=question,
        k=20  # Get more initially to sort by version
    )

    # Filter by partner and sort by version DESC
    filtered = [
        (doc, score) for doc, score in results
        if doc.metadata.get("partner") == partner
    ]
    sorted_chunks = sorted(filtered, key=lambda x: -x[0].metadata.get("version", 0))

    # Return top-k
    return sorted_chunks[:k]



# Step 2: Search
# results = search_relevant_chunks(query, partner="codecraft")

# for doc, score in results:
#     print(f"Version: {doc.metadata['version']} | Score: {score:.2f}")
#     print(doc.page_content[:300])  # Preview first 300 chars
#     print('-' * 40)


# Assumes `search_relevant_chunks` already exists and works
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def generate_prioritized_answer(question, partner, k=8, model_name="gpt-4", temperature=0.0):
    # Step 1: Get top-k chunks for the question
    results = search_relevant_chunks(question, partner=partner, k=k)

    if not results:
        return "Sorry, I couldn't find relevant information in the contract documents."
    expanded_query = expand_query_with_llm(question, results)
    final_results = search_relevant_chunks(expanded_query, partner, k=8)
    print("=============== Final vector search results ================")
    print(final_results)
    print("=============== Final vector search results ================")

    # Step 2: Concatenate context text with most recent versions first
    sorted_context = sorted(
        [doc for doc, _ in final_results],
        key=lambda d: -d.metadata.get("version", 0)
    )
    combined_context = "\n".join([doc.page_content for doc in sorted_context])

    # Step 3: Prompt instruction for merging content with precedence rules
    prompt = f"""
You are a legal assistant helping answer questions based on contract documents.

The documents are organized by version, with version 1 being the main contract and higher numbers being newer addendums.
When clauses conflict, always prefer the **most recent version**.
Note: If any section is explicitly marked as "no longer required", treat its previous content as void and do not use it to answer the question.
Your task is to answer the following question using only the provided contract content, merging any relevant clauses into a single clear and final answer. Do not mention which version a clause comes from. Just give a unified answer as if it was a final merged contract.

If the answer is not clearly found, say: "Not available in contract."

---

Question:
{question}

---

Contract Content (most recent first):
{combined_context}
    """.strip()

    # Step 4: Call the LLM
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    print("================== Prompt ================")
    print(prompt)
    print("================== Prompt ================")
    response = llm([HumanMessage(content=prompt)])

    print("========== Answer ============")
    return response.content

def expand_query_with_llm(original_query, retrieved_chunks, model_name="gpt-4"):
    context = "\n".join([doc.page_content for doc, _ in retrieved_chunks])
    prompt = f"""
You are helping expand a legal question. Given the original question and some partial contract context, rephrase the question to include relevant clauses or sections that may affect the answer.

Original Question: {original_query}

Contract Snippets:
{context}

Expanded Question:
"""
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    response = llm([HumanMessage(content=prompt)])
    ret = response.content.strip()
    print ('========= Expanded Query =============')
    print(ret)
    print ('========= Expanded Query =============')
    return ret


question = "Do developers need to sign non-disclosure agreement?"
partner="AcmeCorp"
print(generate_prioritized_answer(question, partner))