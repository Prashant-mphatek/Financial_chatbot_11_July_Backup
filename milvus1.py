import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections

# Configuration
PDF_FILE = "/home/ubuntu/chatbot_project/Financial_chatbot/Fundamental-of-Banking-English.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "BankingBook_Fundamentals"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Step 1: Load PDF
print("Loading PDF...")
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()
print(f"Loaded {len(docs)} pages")

# Step 2: Split into Chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Step 3: Create Embeddings
print("Creating embeddings...")
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Step 4: Connect to Milvus and Embed
print("Connecting to Milvus...")
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

print("Creating Milvus collection and embedding...")
Milvus.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    drop_old=True  # Drop old collection with same name if exists
)

print(f"✅ Done! Collection '{COLLECTION_NAME}' created with {len(chunks)} chunks.")

