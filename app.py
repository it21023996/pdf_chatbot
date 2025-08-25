import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

# ===== Initialize session state =====
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "db" not in st.session_state:
    st.session_state["db"] = None
if "last_file_name" not in st.session_state:
    st.session_state["last_file_name"] = None
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

# Title and description
st.title("📄 PDF/TXT Q&A Chatbot")
st.write("Upload a PDF or TXT file, then ask questions about it.")

# File uploader
uploaded_file = st.file_uploader("Drag and drop a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Reset everything if a new file is uploaded
    if st.session_state["last_file_name"] != uploaded_file.name:
        st.session_state["messages"] = [
            {"role": "system",
             "content": "You are a helpful assistant. Answer the user's question concisely in 2-3 sentences. Do not repeat the document content."}
        ]
        st.session_state["query_input"] = ""
        st.session_state["last_file_name"] = uploaded_file.name

        # Clear previous vector DB safely
        st.session_state["db"] = None

    # Extract text from file
    if uploaded_file.type == "application/pdf":
        pdf = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf.pages])
    else:
        text = str(uploaded_file.read(), "utf-8")

    st.write("✅ File uploaded successfully!")
    st.write(text[:500] + "...")  # preview first 500 characters

    # Create a new in-memory vector DB for this document
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state["db"] = Chroma.from_texts(
        docs,
        embedding=embeddings,
        collection_name=f"session_{uuid.uuid4().hex}",
        persist_directory=None  # in-memory DB, not saved to disk
    )

    st.write("✅ Vector store created")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to handle question input
def ask_question():
    query = st.session_state["query_input"].strip()
    if query and st.session_state["db"] is not None:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": query})

        # Retrieve relevant documents
        retriever = st.session_state["db"].as_retriever(search_kwargs={"k": 5})
        results = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in results])

        # Create prompt
        prompt = f"""
You are a helpful assistant. Answer the user's question concisely in 2-3 sentences.
Do not repeat the document content.

Context:
{context}

Question: {query}
Answer:
"""

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": st.session_state["messages"][0]["content"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        # Add assistant message
        output_text = response.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": output_text})

        # Clear input box
        st.session_state["query_input"] = ""

# Display chat history
for msg in st.session_state.get("messages", [])[1:]:
    if msg["role"] == "user":
        st.markdown(f"👤 **Question :** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"🤖 **Answer :** {msg['content']}")

# Question input at the bottom like ChatGPT
st.text_input("Ask a question about the document:", key="query_input", on_change=ask_question)

