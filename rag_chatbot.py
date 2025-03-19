import streamlit as st
import os
import json
import hashlib
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import tempfile

# App title and configuration
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚")
st.title("PDF-Powered RAG Chatbot")

# Create directory for saved sessions if it doesn't exist
SAVE_DIR = os.path.join(os.getcwd(), "saved_sessions")
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize session state for storing conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "sessions_metadata" not in st.session_state:
    # Load existing sessions metadata
    metadata_path = os.path.join(SAVE_DIR, "sessions_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            st.session_state.sessions_metadata = json.load(f)
    else:
        st.session_state.sessions_metadata = {}

# Sidebar for OpenAI API key and document upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    os.environ["OPENAI_API_KEY"] = api_key

    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    model_option = st.selectbox(
        "Select OpenAI Model",
        ("gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o")
    )

    process_button = st.button("Process Documents")

    # Add Session Management UI
    st.header("Session Management")

    # Save current session
    session_name = st.text_input("Session Name", placeholder="Enter a name for this session")
    save_button = st.button("Save Current Session")

    if save_button and session_name:
        save_session(session_name)
        st.success(f"Session '{session_name}' saved successfully!")

    # Load existing session
    st.subheader("Load Existing Session")
    session_options = list(st.session_state.sessions_metadata.keys())

    if session_options:
        selected_session = st.selectbox("Select a session to load", session_options)
        load_button = st.button("Load Selected Session")

        if load_button and selected_session:
            load_session(selected_session)
            st.success(f"Session '{selected_session}' loaded successfully!")
            st.rerun()  # Rerun the app to update UI with loaded session
    else:
        st.info("No saved sessions found.")

    # Delete session option
    if session_options:
        st.subheader("Delete Session")
        delete_session = st.selectbox("Select a session to delete", session_options, key="delete_selector")
        delete_button = st.button("Delete Selected Session")

        if delete_button and delete_session:
            delete_saved_session(delete_session)
            st.success(f"Session '{delete_session}' deleted successfully!")
            st.rerun()


# Function to save session
def save_session(session_name):
    # Generate a unique session ID if one doesn't exist
    if not st.session_state.session_id:
        st.session_state.session_id = hashlib.md5(f"{session_name}_{datetime.now().isoformat()}".encode()).hexdigest()

    session_id = st.session_state.session_id
    session_dir = os.path.join(SAVE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save vector store if it exists
    if st.session_state.vector_store:
        vector_store_path = os.path.join(session_dir, "vector_store")
        st.session_state.vector_store.save_local(vector_store_path)

    # Save conversation messages
    messages_path = os.path.join(session_dir, "messages.json")
    with open(messages_path, "w") as f:
        json.dump(st.session_state.messages, f)

    # Save processed files info
    files_path = os.path.join(session_dir, "processed_files.json")
    with open(files_path, "w") as f:
        json.dump(st.session_state.processed_files, f)

    # Update metadata
    st.session_state.sessions_metadata[session_name] = {
        "id": session_id,
        "created": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "document_count": len(st.session_state.processed_files),
        "message_count": len(st.session_state.messages)
    }

    # Save metadata
    metadata_path = os.path.join(SAVE_DIR, "sessions_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(st.session_state.sessions_metadata, f)


# Function to load session
def load_session(session_name):
    if session_name not in st.session_state.sessions_metadata:
        st.error(f"Session '{session_name}' not found.")
        return

    session_id = st.session_state.sessions_metadata[session_name]["id"]
    session_dir = os.path.join(SAVE_DIR, session_id)

    # Load messages
    messages_path = os.path.join(session_dir, "messages.json")
    if os.path.exists(messages_path):
        with open(messages_path, "r") as f:
            st.session_state.messages = json.load(f)

    # Load processed files info
    files_path = os.path.join(session_dir, "processed_files.json")
    if os.path.exists(files_path):
        with open(files_path, "r") as f:
            st.session_state.processed_files = json.load(f)

    # Load vector store if it exists
    vector_store_path = os.path.join(session_dir, "vector_store")
    if os.path.exists(vector_store_path):
        if api_key:  # Need API key to create embeddings
            try:
                embeddings = OpenAIEmbeddings()
                st.session_state.vector_store = FAISS.load_local(vector_store_path, embeddings)

                # Recreate conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
        else:
            st.warning("Please enter your OpenAI API key to load the vector store.")

    # Update session ID
    st.session_state.session_id = session_id

    # Update metadata
    st.session_state.sessions_metadata[session_name]["last_modified"] = datetime.now().isoformat()
    metadata_path = os.path.join(SAVE_DIR, "sessions_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(st.session_state.sessions_metadata, f)


# Function to delete a saved session
def delete_saved_session(session_name):
    if session_name not in st.session_state.sessions_metadata:
        st.error(f"Session '{session_name}' not found.")
        return

    session_id = st.session_state.sessions_metadata[session_name]["id"]
    session_dir = os.path.join(SAVE_DIR, session_id)

    # Delete directory
    if os.path.exists(session_dir):
        import shutil
        shutil.rmtree(session_dir)

    # Update metadata
    del st.session_state.sessions_metadata[session_name]
    metadata_path = os.path.join(SAVE_DIR, "sessions_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(st.session_state.sessions_metadata, f)


# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    processed_files = []

    for pdf in pdf_files:
        # Create a temporary file to handle the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name

        try:
            pdf_reader = PdfReader(temp_file_path)
            file_text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                file_text += page_text + "\n"

            text += file_text

            # Store processed file info
            file_hash = hashlib.md5(file_text.encode()).hexdigest()
            processed_files.append({
                "name": pdf.name,
                "pages": len(pdf_reader.pages),
                "hash": file_hash,
                "processed_date": datetime.now().isoformat()
            })

        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    return text, processed_files


# Function to create vector store from text
def create_vector_store(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    if not api_key:
        st.warning("Please enter your OpenAI API key to enable embeddings.")
        return None

    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None


# Function to create conversation chain
def get_conversation_chain(vector_store):
    if not api_key:
        return None

    try:
        llm = ChatOpenAI(model=model_option, temperature=0.0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # If we have existing messages, populate the memory
        if st.session_state.messages:
            # Convert flat messages to the format expected by ConversationBufferMemory
            from langchain.schema import HumanMessage, AIMessage
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    memory.chat_memory.add_ai_message(msg["content"])

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None


# Process uploaded documents
if process_button and uploaded_files:
    with st.spinner("Processing PDFs..."):
        # Extract text from PDFs
        extracted_text, new_processed_files = extract_text_from_pdfs(uploaded_files)

        if extracted_text:
            # Update processed files
            st.session_state.processed_files.extend(new_processed_files)

            # Create vector store
            new_vector_store = create_vector_store(extracted_text)

            if new_vector_store:
                if st.session_state.vector_store:
                    # If we already have a vector store, merge with new one
                    # First, we get all the documents from our new store
                    new_docs = []
                    new_doc_ids = list(new_vector_store.docstore._dict.keys())
                    for doc_id in new_doc_ids:
                        new_docs.append(new_vector_store.docstore.search(doc_id))

                    # Then we add them to our existing store
                    existing_embedding_function = st.session_state.vector_store._embedding_function
                    st.session_state.vector_store.add_documents(new_docs,
                                                                embedding_function=existing_embedding_function)
                else:
                    # If no existing vector store, just use the new one
                    st.session_state.vector_store = new_vector_store

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
        else:
            st.warning("No text could be extracted from the uploaded documents.")

# Show processed documents
if st.session_state.processed_files:
    with st.expander("Processed Documents"):
        for doc in st.session_state.processed_files:
            st.write(f"📄 {doc['name']} - {doc['pages']} pages")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if API key is provided
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    # Check if documents are processed
    elif st.session_state.conversation is None:
        st.warning("Please upload and process documents first.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                result = st.session_state.conversation({"question": prompt})
                response = result["answer"]

                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Auto-save session if there's an existing session id
                if st.session_state.session_id and st.session_state.sessions_metadata:
                    # Find session name from id
                    session_name = None
                    for name, metadata in st.session_state.sessions_metadata.items():
                        if metadata["id"] == st.session_state.session_id:
                            session_name = name
                            break

                    if session_name:
                        save_session(session_name)

            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Additional information in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Enter your OpenAI API key
    2. Upload PDF documents
    3. Click 'Process Documents'
    4. Ask questions about the content
    5. Save your session to continue later
    """)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) to:
    - Extract text from uploaded PDFs
    - Create embeddings of document chunks
    - Retrieve relevant context from your documents
    - Generate responses based on the document content
    - Save and restore your learning sessions
    """)