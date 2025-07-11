import logging
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
import requests
import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

st.set_page_config(page_title="Banking Assistant", layout="centered")
st.title("\U0001F3E6 Banking Chatbot")

# Simple login
temp_credentials = {"12345": "pass123", "67890": "pass678", "11111": "pass111"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    with st.sidebar:
        st.header("\U0001F512 Login Required")
        acc = st.text_input("Account No")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if acc in temp_credentials and temp_credentials[acc] == pwd:
                st.session_state.logged_in = True
                st.session_state.accountno = acc
                st.rerun()
            else:
                st.error("❌ Invalid Credentials")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_intent_classifier():
    path = "/home/ubuntu/model_files"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    connections.connect(host="localhost", port="19530")
    if utility.has_collection("BankingRAGCollection_Omnispay"):
        return Milvus(embedding_function=embeddings, collection_name="BankingRAGCollection_Omnispay",
                      connection_args={"host": "localhost", "port": "19530"}).as_retriever(search_kwargs={"k": 3})
    return None

@st.cache_resource
def load_qa_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

def get_balance():
    acc = st.session_state.accountno
    try:
        r = requests.get("http://localhost:8082/balance", params={"accountno": acc})
        if r.status_code == 200:
            data = r.json()
            if "balance" in data:
                return f"Your current balance is ₹{data['balance']:.2f}."
            else:
                return "⚠️ Balance not found in response."
        return "\u274C API error while fetching balance."
    except Exception as e:
        return f"\u274C Unable to fetch account balance: {e}"

def get_transactions():
    acc = st.session_state.accountno
    try:
        r = requests.get("http://localhost:8082/transactions", params={"accountno": acc})
        if r.status_code == 200:
            txns = r.json().get('transactions', [])
            if not txns:
                return "⚠️ No transactions found."
            return "Recent transactions:\n" + "\n".join(
                [f"{x['desc']} → ₹{abs(x['amount'])} ({x['type']})" for x in txns]
            )
        return "\u274C API error while fetching transactions."
    except Exception as e:
        return f"\u274C Unable to fetch transactions: {e}"


@st.cache_resource
def get_chain():
    retriever = load_vectorstore()
    hf_pipeline = load_qa_pipeline()
    hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant for a financial services platform. Use only the context below to answer the question.
If the answer is not explicitly in the context, respond: \"I'm sorry, the information is not available.\"

Context:
{context}

Question: {question}

Answer:
"""
    )
    return ConversationalRetrievalChain.from_llm(
        llm=hf_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

st.sidebar.markdown("### \U0001F4C2 Upload Knowledge Files")
uploaded_files = st.sidebar.file_uploader("Upload .txt, .pdf, .docx, .xlsx files", type=["txt", "pdf", "docx", "xlsx"], accept_multiple_files=True)
if uploaded_files:
    raw_docs = []
    os.makedirs("temp_uploads", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("temp_uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        else:
            loader = TextLoader(file_path)
        raw_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    store = Milvus.from_documents(
        chunks,
        embedding,
        collection_name="BankingRAGCollection_Omnispay",
        connection_args={"host": "localhost", "port": "19530"},
        drop_old=False
    )
    st.sidebar.success("✅ Files uploaded and embedded.")

# Show chat history
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="\U0001F464"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="\U0001F916"):
        st.markdown(bot_msg)

# Query input
query = st.chat_input("Ask your banking query here (English only)...")
if query:
    with st.chat_message("user", avatar="\U0001F464"):
        st.markdown(query)

    progress = st.progress(0, text="\U0001F50D Understanding your query...")
    start = time.time()
    progress.progress(40, text="\U0001F4CA Classifying intent...")

    intent_classifier = load_intent_classifier()
    intent = intent_classifier(query)[0]['label'].lower()
    st.sidebar.markdown(f"**🧠 Detected Intent:** `{intent}`")

    try:
        query_lc = query.lower()
        is_transaction_query = any(word in query_lc for word in ["transaction", "transactions", "recent", "statement", "history"])
        is_balance_query = "balance" in query_lc

        if intent == "data retrieval question" and (is_balance_query or is_transaction_query):
            progress.progress(70, text="\U0001F4B3 Retrieving account data...")
            response = get_balance() if is_balance_query else get_transactions()
            source_docs = []
        else:
            progress.progress(70, text="\U0001F50E Searching knowledge base...")
            chain = get_chain()
            result = chain.invoke({"question": query})
            response = result["answer"]
            source_docs = result.get("source_documents", [])
    except Exception as e:
        response = f"❌ Error: {str(e)}"
        source_docs = []


    end = time.time()
    progress.progress(100, text="✅ Done in {:.2f} seconds".format(end - start))
    time.sleep(0.2)
    progress.empty()

    with st.chat_message("assistant", avatar="\U0001F916"):
        st.markdown(f"**Answer (in {end-start:.2f}s):**")
        st.markdown(response)
        if source_docs:
            with st.expander("📄 Retrieved Context"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Doc {i+1}:**")
                    st.markdown(doc.page_content[:800])

    st.session_state.chat_history.append((query, response))

