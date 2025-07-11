import logging
logging.getLogger("pymilvus").setLevel(logging.CRITICAL)

import streamlit as st
import time
import torch
import os
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Debug GPU info
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

st.set_page_config(page_title="Banking Assistant", layout="centered")
st.title("\U0001F3E6 Banking Chatbot")

credentials = {"12345": "pass123", "67890": "pass678", "11111": "pass111"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    with st.sidebar:
        st.header("\U0001F512 Login Required")
        acc = st.text_input("Account No")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if acc in credentials and credentials[acc] == pwd:
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
                      connection_args={"host": "localhost", "port": "19530"}).as_retriever(search_kwargs={"k": 6})
    return None

#@st.cache_resource
#def load_translation():
#    return (
#        pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en"),
#        pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
#    )

@st.cache_resource
def load_translation():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch

    model_name = "facebook/nllb-200-distilled-600M"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer normally
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ DO NOT use device_map or low_cpu_mem_usage
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True  # force use of safetensors instead of legacy PyTorch
    )

    # ✅ Now manually move it to CUDA
    model = model.to(device)

    # ✅ Pipeline will use model already on device
    ar_to_en_pipeline = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang="arb",
        tgt_lang="eng_Latn",
        device=0 if torch.cuda.is_available() else -1
    )

    en_to_ar_pipeline = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang="eng_Latn",
        tgt_lang="arb",
        device=0 if torch.cuda.is_available() else -1
    )

    return ar_to_en_pipeline, en_to_ar_pipeline


@st.cache_resource
def load_qa_pipeline():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@st.cache_resource
def get_chain():
    retriever = load_vectorstore()
    hf_pipeline = load_qa_pipeline()
    hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an AI assistant for a financial services platform. Use only the context below to answer the question.
If the answer is not explicitly in the context, respond: "I'm sorry, the information is not available."

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

# File upload section
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
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

# Chat history
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="\U0001F464"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="\U0001F916"):
        st.markdown(bot_msg)

# Query processing
query = st.chat_input("Ask your banking query here...")
if query:
    with st.chat_message("user", avatar="\U0001F464"):
        st.markdown(query)

    progress = st.progress(0, text="\U0001F50D Understanding your query...")
    start = time.time()
    lang = detect(query)
    progress.progress(20, text="\U0001F310 Translating if needed...")
    ar_to_en, en_to_ar = load_translation()
    if lang == "ar":
        query = ar_to_en(query)[0]['translation_text']
    progress.progress(40, text="\U0001F4CA Classifying intent...")
    intent_classifier = load_intent_classifier()
    intent = intent_classifier(query)[0]['label'].lower()
    st.sidebar.markdown(f"**🧠 Detected Intent:** `{intent}`")

    try:
        if intent == "data retrieval question" and any(w in query.lower() for w in ["balance", "transaction", "history"]):
            progress.progress(70, text="\U0001F4B3 Retrieving account data...")
            response = "Account data feature not implemented in this demo."
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

    if lang == "ar":
        progress.progress(90, text="\U0001F310 Translating answer to Arabic...")
        response = en_to_ar(response)[0]['translation_text']

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
