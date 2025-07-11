import torch
import logging
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
import requests
import streamlit as st
import time
import os
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ========= Streamlit Config ==========
st.set_page_config(page_title="Banking Assistant", layout="centered")
st.title("\U0001F3E6 Banking Chatbot")

# ========= Simple Login ==========
temp_credentials = {"12345": "pass123", "67890": "pass678", "11111": "pass111"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    with st.sidebar:
        st.header("\U0001F510 Login Required")
        acc = st.text_input("Account No")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if acc in temp_credentials and temp_credentials[acc] == pwd:
                st.session_state.logged_in = True
                st.session_state.accountno = acc
                st.rerun()
            else:
                st.error("\u274C Invalid Credentials")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========= Translation Functions ==========
def is_arabic(text):
    return any('\u0600' <= c <= '\u06FF' for c in text)

def translate_query(query):
    if is_arabic(query):
        translated = GoogleTranslator(source='auto', target='en').translate(query)
        return translated, 'ar'
    return query, 'en'

def translate_response(response, lang):
    if lang == 'ar':
        return GoogleTranslator(source='en', target='ar').translate(response)
    return response

# ========= Loaders ==========
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
def load_ollama_llm():
    return Ollama(model="llama3.2", temperature=0.6)

@st.cache_resource
def get_chain():
    retriever = load_vectorstore()
    ollama_llm = load_ollama_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
You are OmnispayBot, a helpful assistant for Omnispay.

Use the conversation history to understand and answer follow-up questions. 
Only use the context provided below from Omnispay's documentation. 
If the answer is not explicitly in the context, say:
\"I'm sorry, I couldn't find that information in Omnispay's documentation.\"

---
Conversation History:
{chat_history}

---
Context:
{context}

---
User Question:
{question}

---
Answer:
"""
    )

    return ConversationalRetrievalChain.from_llm(
        llm=ollama_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ========= API Functions ==========
def get_balance():
    acc = st.session_state.accountno
    try:
        r = requests.get("http://localhost:8082/balance", params={"accountno": acc})
        if r.status_code == 200:
            data = r.json()
            if "balance" in data:
                return f"Your current balance is \u20B9{data['balance']:.2f}."
            else:
                return "\u26A0\ufe0f Balance not found in response."
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
                return "\u26A0\ufe0f No transactions found."
            return "Recent transactions:\n" + "\n".join(
                [f"{x['desc']} \u2192 \u20B9{abs(x['amount'])} ({x['type']})" for x in txns]
            )
        return "\u274C API error while fetching transactions."
    except Exception as e:
        return f"\u274C Unable to fetch transactions: {e}"

# ========= File Upload ==========
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
    st.sidebar.success("\u2705 Files uploaded and embedded.")

# ========= Chat Display ==========
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="\U0001F464"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="\U0001F916"):
        st.markdown(bot_msg)

# ========= Chat Input Handling ==========
query = st.chat_input("Ask your banking query here (English or Arabic)...")
if query:
    with st.chat_message("user", avatar="\U0001F464"):
        st.markdown(query)

    progress = st.progress(0, text="\U0001F310 Translating query if needed...")
    translated_query, detected_lang = translate_query(query)

    progress.progress(20, text="\U0001F50D Classifying intent...")
    intent_classifier = load_intent_classifier()
    intent = intent_classifier(translated_query)[0]['label'].lower()

    st.sidebar.markdown(f"**\U0001F310 Detected Language:** `{detected_lang}`")
    st.sidebar.markdown(f"**\U0001F9E0 Intent:** `{intent}`")

    try:
        query_lc = translated_query.lower()
        is_transaction_query = any(word in query_lc for word in ["transaction", "transactions", "recent", "statement", "history"])
        is_balance_query = "balance" in query_lc

        if intent == "data retrieval question" and (is_balance_query or is_transaction_query):
            progress.progress(60, text="\U0001F4B3 Getting account info...")
            response = get_balance() if is_balance_query else get_transactions()
            source_docs = []
        else:
            progress.progress(60, text="\U0001F4DA Retrieving answer from knowledge base...")
            chain = get_chain()
            result = chain.invoke({"question": translated_query})
            response = result["answer"]
            source_docs = result.get("source_documents", [])
    except Exception as e:
        response = f"\u274C Error: {str(e)}"
        source_docs = []

    response = translate_response(response, detected_lang)

    progress.progress(100, text="\u2705 Done")
    time.sleep(0.3)
    progress.empty()

    with st.chat_message("assistant", avatar="\U0001F916"):
        st.markdown(response)
        if source_docs:
            with st.expander("\U0001F4C4 Retrieved Context"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Doc {i+1}:**")
                    st.markdown(doc.page_content[:800])

    st.session_state.chat_history.append((query, response))
