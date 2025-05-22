import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPEN_AI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

import re
from collections import defaultdict

BASE_DIR = '/Users/soumya/Technicals/pythonProject/GeminiGenAI/contract'
def load_documents_by_partner(files):
    partner_docs = defaultdict(list)
    for f in files:
        match = re.match(r'(\w+)_(\d+)\.txt', f)
        if match:
            partner, number = match.groups()
            with open(os.path.join(BASE_DIR, f), 'r') as f:
                partner_docs[partner].append((int(number), f.read()))

    # Sort each partner's docs by version
    for partner in partner_docs:
        partner_docs[partner].sort(key=lambda x: x[0])
    return partner_docs


splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)

def build_chunks(partner_docs):
    chunks = []
    for partner, docs in partner_docs.items():
        for version, content in docs:
            for chunk in splitter.split_text(content):
                chunks.append({
                    "partner": partner,
                    "version": version,
                    "text": chunk
                })
    return chunks



def store_chunks_in_chroma(chunks, persist_directory="./chroma_db"):
    # Convert chunks to LangChain Document format
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "partner": chunk["partner"],
                "version": chunk["version"]
            }
        )
        for chunk in chunks
    ]

    # Initialize OpenAI embedding model
    embedding_model = OpenAIEmbeddings()  # OR: AzureOpenAIEmbeddings / HuggingFaceEmbeddings

    # Create and persist Chroma store
    vectordb = Chroma.from_documents(
        documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Stored {len(documents)} chunks in ChromaDB at '{persist_directory}'")

files = txt_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.txt')]
PARTNER_DOCS = load_documents_by_partner(files)
chunks = build_chunks(PARTNER_DOCS)
store_chunks_in_chroma(chunks)