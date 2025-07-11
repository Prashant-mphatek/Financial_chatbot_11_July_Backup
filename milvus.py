# Simple RAG System for Ancient Greece Q&A

import os
import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections

# Configuration
DATA_DIR = "/home/ubuntu/chatbot_project/Financial_chatbot"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3.2"
LLM_BASE_URL = "http://localhost:11434"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "BankingRAGCollection_Omnispay"

# Step 1: Load Documents
print("Loading documents...")
loader = DirectoryLoader(
    DATA_DIR,
    glob='**/*.txt',
    loader_cls=TextLoader
)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Step 2: Split Documents into Chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Step 3: Create Embeddings and Vector Store
print("Creating embeddings and vector store...")
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Connect to Milvus
print("Connecting to Milvus...")
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Create Milvus vector store
vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embedding,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    collection_name=COLLECTION_NAME,
    drop_old=True  # Remove existing collection if it exists
)
print("Milvus vector store created!")
