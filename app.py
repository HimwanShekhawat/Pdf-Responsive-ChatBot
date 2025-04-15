import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
import chromadb
from chromadb.config import Settings


#  Directories
DOCS_DIR = "docs"
DB_DIR = "db"

MODEL_PATH = "/app/LaMini-T5-61M"  

device = "cuda" if torch.cuda.is_available() else "cpu"

#  Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)

#  Load the text generation pipeline
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    return HuggingFacePipeline(pipeline=pipe)

#  Load retrieval-based QA system
@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

qa = qa_llm()

#  Function to process PDF and create embeddings
def ingest_pdf(uploaded_file):
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    documents = PDFMinerLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(settings=Settings(persist_directory=DB_DIR))
    collection = client.get_or_create_collection(name="pdf_documents")

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    for i, text in enumerate(texts):
        if str(i) not in existing_ids:  # Avoid duplicate inserts
            collection.add(
                ids=[str(i)],
                documents=[text.page_content],
                metadatas=[text.metadata],
            )

    st.success("✅ PDF processed and stored for searching!")

#  Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

#  **Streamlit UI - Chatbot**
st.markdown(
    """
    <style>
        .chat-container {
            max-width: 700px;
            margin: auto;
        }
        .chat-message {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .chat-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 24px;
            font-weight: bold;
            background-color: #f1f1f1;
        }
        .chat-bubble {
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 60%;
            word-wrap: break-word;
        }
        .bot-message {
            background-color: #E0E0E0;
            align-self: flex-start;
        }
        .user-message {
            background-color: #DCF8C6;
            align-self: flex-end;
            margin-left: auto;
        }
        .input-container {
            display: flex;
            align-items: center;
            border: 1px solid #ccc;
            border-radius: 20px;
            padding: 5px 10px;
            background: white;
            margin-top: 10px;
        }
        .input-box {
            border: none;
            outline: none;
            flex: 1;
            font-size: 16px;
            padding: 5px;
        }
        .send-btn {
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            color: #0078FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 PDF Chatbot")

#  **PDF Upload Section**
uploaded_file = st.file_uploader("📄 Upload a PDF for Q&A", type=["pdf"])
if uploaded_file is not None:
    ingest_pdf(uploaded_file)

#  **Display Chat History**
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    role, text, avatar = msg
    alignment_class = "user-message" if role == "User" else "bot-message"
    
    st.markdown(
        f"""
        <div class="chat-message">
            <div class="chat-avatar">{avatar}</div>
            <div class="chat-bubble {alignment_class}">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

#  **User Input Section with Right-Headed Arrow**
col1, col2 = st.columns([8, 1])

with col1:
   
    user_input = st.text_input("" ,placeholder="Type your message...", key="chat_input", label_visibility="collapsed")

with col2:
    send_pressed = st.button("➤", key="send_button", help="Send Message")

#  Handle Message Sending (Enter key or Send button)
def send_message():
    if user_input.strip():
        try:
            response_data = qa(user_input)
            bot_response = response_data["result"]
        except Exception as e:
            bot_response = "⚠️ Error generating response."

        # Add messages to session state (User & Bot)
        st.session_state["messages"].append(("User", user_input, "👨‍💼"))
        st.session_state["messages"].append(("Bot", bot_response, "🤖"))

        # Refresh chat UI
        st.rerun()

#  **Trigger send_message() on Enter key or Send button click**
if send_pressed or (user_input and st.session_state.get("chat_input_submit", False)):
    send_message()