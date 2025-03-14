import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import json
from datetime import datetime

# Set page config
st.set_page_config(page_title="Document Chat Assistant", page_icon="📚")

# Initialize session state variables if they don't exist
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/vector_stores"):
    os.makedirs("data/vector_stores")
if not os.path.exists("data/conversations"):
    os.makedirs("data/conversations")

# Main header
st.header("📚 Chat with your PDFs")

# Sidebar for API key, document upload, and saved sessions
with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    os.environ["OPENAI_API_KEY"] = api_key

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

    knowledge_base_name = st.text_input("Knowledge Base Name", "my_knowledge_base")
    process_button = st.button("Process Documents")

    # Load saved vector stores
    st.subheader("Saved Knowledge Bases")
    saved_vector_stores = [f.replace(".faiss", "") for f in os.listdir("data/vector_stores") if f.endswith(".faiss")]
    if saved_vector_stores:
        selected_vs = st.selectbox("Select a saved knowledge base", saved_vector_stores)
        load_vs_button = st.button("Load Knowledge Base")

        if load_vs_button and selected_vs:
            with st.spinner(f"Loading knowledge base {selected_vs}..."):
                # Load the vector store
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.load_local(f"data/vector_stores/{selected_vs}", embeddings)
                st.session_state.vector_store = vector_store

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.session_state.processed_docs = True
                st.sidebar.success(f"Knowledge base {selected_vs} loaded successfully!")
    else:
        st.info("No saved knowledge bases found.")

    # Load saved conversations
    st.subheader("Saved Conversations")
    saved_conversations = [f.replace(".json", "") for f in os.listdir("data/conversations") if f.endswith(".json")]
    if saved_conversations:
        selected_conv = st.selectbox("Select a saved conversation", saved_conversations)
        load_conv_button = st.button("Load Conversation")

        if load_conv_button and selected_conv:
            with st.spinner(f"Loading conversation {selected_conv}..."):
                with open(f"data/conversations/{selected_conv}.json", "r") as f:
                    st.session_state.chat_history = json.load(f)
                st.sidebar.success(f"Conversation {selected_conv} loaded successfully!")
    else:
        st.info("No saved conversations found.")

    # Save current conversation
    if st.session_state.chat_history:
        save_conv_name = st.text_input("Save current conversation as",
                                       f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if st.button("Save Current Conversation"):
            with open(f"data/conversations/{save_conv_name}.json", "w") as f:
                json.dump(st.session_state.chat_history, f)
            st.sidebar.success(f"Conversation saved as {save_conv_name}!")


# Function to extract text from PDFs
def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# Function to create vector store
def create_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


# Function to create conversation chain
def get_conversation_chain(vector_store):
    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Process documents when button is clicked
if process_button and uploaded_files and api_key:
    with st.spinner("Processing documents..."):
        # Extract text from PDFs
        raw_text = extract_pdf_text(uploaded_files)

        # Split text into chunks
        text_chunks = split_text_into_chunks(raw_text)
        st.sidebar.info(f"Documents split into {len(text_chunks)} chunks")

        # Create vector store
        vector_store = create_vector_store(text_chunks)
        st.session_state.vector_store = vector_store

        # Save vector store
        vector_store.save_local(f"data/vector_stores/{knowledge_base_name}")

        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vector_store)
        st.session_state.processed_docs = True
        st.sidebar.success(f"Documents processed and saved as '{knowledge_base_name}'!")

# Chat interface
st.subheader("Chat")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
elif not st.session_state.processed_docs:
    st.info("Upload your documents and click 'Process Documents' or load a saved knowledge base to start chatting.")
else:
    # Chat input and response
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history.append({"user": user_question, "bot": response["answer"]})

    # Display chat history
    for message in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {message['user']}")
        st.markdown(f"**Assistant:** {message['bot']}")
        st.divider()

# Display useful information in the expander
with st.expander("How to use this app"):
    st.markdown("""
    1. Enter your OpenAI API key in the sidebar
    2. Upload PDF documents in the sidebar
    3. Give your knowledge base a name and click 'Process Documents'
    4. Ask questions about the content of your documents
    5. Save conversations for future reference
    6. Load previously created knowledge bases without reprocessing documents

    **Note:** This app uses OpenAI's API to process your documents and generate responses, which may incur charges to your OpenAI account based on usage.
    """)