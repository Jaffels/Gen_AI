import streamlit as st
import os
import json
import hashlib
import sqlite3
import base64
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import tempfile


# Set your OpenAI API Key directly in the code
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# App title and configuration
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("University Questions Chatbot")

# Create directory for saved sessions if it doesn't exist
SAVE_DIR = os.path.join(os.getcwd(), "saved_sessions")
os.makedirs(SAVE_DIR, exist_ok=True)

# Database setup
DB_PATH = os.path.join(os.getcwd(), "rag_chatbot.db")

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for PDF documents
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        content BLOB NOT NULL,
        pages INTEGER NOT NULL,
        upload_date TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        processed BOOLEAN DEFAULT FALSE
    )
    ''')
    
    # Create table for text chunks
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        content TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )
    ''')
    
    # Create table for embeddings
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL,
        embedding BLOB NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    ''')
    
    # Create table for sessions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created TEXT NOT NULL,
        last_modified TEXT NOT NULL,
        vector_store_path TEXT
    )
    ''')
    
    # Create table for session_documents (many-to-many relationship)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS session_documents (
        session_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        PRIMARY KEY (session_id, document_id),
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )
    ''')
    
    # Create table for messages
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Database functions
def save_pdf_to_db(file_obj, filename):
    """Save a PDF file to the database"""
    # Reset file pointer to beginning
    file_obj.seek(0)
    
    # Read file content as binary
    pdf_content = file_obj.read()
    file_size = len(pdf_content)
    
    # Create temporary file to read with PyPDF2
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_content)
        temp_file_path = temp_file.name
    
    # Get page count
    try:
        pdf_reader = PdfReader(temp_file_path)
        page_count = len(pdf_reader.pages)
    except Exception as e:
        os.unlink(temp_file_path)
        st.error(f"Error reading PDF: {str(e)}")
        return None
    finally:
        os.unlink(temp_file_path)
    
    # Generate document ID
    doc_id = hashlib.md5(pdf_content).hexdigest()
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if document already exists
    cursor.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO documents (id, name, content, pages, upload_date, file_size, processed) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, filename, pdf_content, page_count, datetime.now().isoformat(), file_size, False)
        )
        conn.commit()
    
    conn.close()
    
    return {
        "id": doc_id,
        "name": filename,
        "pages": page_count,
        "size": file_size,
        "upload_date": datetime.now().isoformat()
    }

def get_document_from_db(doc_id):
    """Retrieve a document from the database by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, content, pages, upload_date, file_size, processed FROM documents WHERE id = ?", (doc_id,))
    doc = cursor.fetchone()
    conn.close()
    
    if doc:
        return {
            "id": doc[0],
            "name": doc[1],
            "content": doc[2],  # Binary content
            "pages": doc[3],
            "upload_date": doc[4],
            "size": doc[5],
            "processed": bool(doc[6])
        }
    return None

def get_all_documents():
    """Get a list of all documents in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, pages, upload_date, file_size, processed FROM documents")
    docs = cursor.fetchall()
    conn.close()
    
    result = []
    for doc in docs:
        result.append({
            "id": doc[0],
            "name": doc[1],
            "pages": doc[2],
            "upload_date": doc[3],
            "size": doc[4],
            "processed": bool(doc[5])
        })
    
    return result

def extract_text_from_db_document(doc_id):
    """Extract text from a document stored in the database"""
    doc = get_document_from_db(doc_id)
    if not doc:
        return ""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(doc["content"])
        temp_file_path = temp_file.name
    
    try:
        pdf_reader = PdfReader(temp_file_path)
        text = ""
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n"
        
        # Mark document as processed
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET processed = TRUE WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from {doc['name']}: {str(e)}")
        return ""
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

def save_chunks_to_db(doc_id, chunks, embeddings_list=None, model_name=None):
    """Save text chunks to the database with optional embeddings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First delete any existing chunks and their embeddings for this document
    # Get all chunk ids for this document
    cursor.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
    chunk_ids = [row[0] for row in cursor.fetchall()]
    
    # Delete associated embeddings
    if chunk_ids:
        placeholders = ','.join(['?'] * len(chunk_ids))
        cursor.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
    
    # Delete the chunks
    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    
    # Insert new chunks and their embeddings
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{doc_id}_{i}_{chunk}".encode()).hexdigest()
        chunk_ids.append(chunk_id)
        cursor.execute(
            "INSERT INTO chunks (id, document_id, content, chunk_index) VALUES (?, ?, ?, ?)",
            (chunk_id, doc_id, chunk, i)
        )
    
    # Insert embeddings if provided
    if embeddings_list and model_name and len(embeddings_list) == len(chunks):
        for chunk_id, embedding_vector in zip(chunk_ids, embeddings_list):
            embedding_id = hashlib.md5(f"{chunk_id}_{model_name}".encode()).hexdigest()
            # Serialize the embedding vector to binary
            embedding_blob = base64.b64encode(json.dumps(embedding_vector).encode())
            cursor.execute(
                "INSERT INTO embeddings (id, chunk_id, embedding, model, created_at) VALUES (?, ?, ?, ?, ?)",
                (embedding_id, chunk_id, embedding_blob, model_name, datetime.now().isoformat())
            )
    
    conn.commit()
    conn.close()

def get_chunks_for_document(doc_id):
    """Get all chunks for a document"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, content, chunk_index FROM chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,))
    chunks = cursor.fetchall()
    conn.close()
    
    return [{"id": chunk[0], "content": chunk[1], "index": chunk[2]} for chunk in chunks]

def get_embeddings_for_chunks(chunk_ids, model_name=None):
    """Get embeddings for specific chunks, optionally filtered by model"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if model_name:
        placeholders = ','.join(['?'] * len(chunk_ids))
        query = f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders}) AND model = ?"
        cursor.execute(query, chunk_ids + [model_name])
    else:
        placeholders = ','.join(['?'] * len(chunk_ids))
        query = f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})"
        cursor.execute(query, chunk_ids)
    
    results = cursor.fetchall()
    conn.close()
    
    embeddings_dict = {}
    for chunk_id, embedding_blob in results:
        # Deserialize the embedding vector from binary
        embedding_json = base64.b64decode(embedding_blob).decode()
        embedding_vector = json.loads(embedding_json)
        embeddings_dict[chunk_id] = embedding_vector
    
    return embeddings_dict

def save_session_to_db(session_name, document_ids, vector_store_path=None):
    """Save a session to the database"""
    # Generate session ID if needed
    if not st.session_state.session_id:
        st.session_state.session_id = hashlib.md5(f"{session_name}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    session_id = st.session_state.session_id
    now = datetime.now().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if session exists
    cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if cursor.fetchone() is None:
        # Create new session
        cursor.execute(
            "INSERT INTO sessions (id, name, created, last_modified, vector_store_path) VALUES (?, ?, ?, ?, ?)",
            (session_id, session_name, now, now, vector_store_path)
        )
    else:
        # Update existing session
        cursor.execute(
            "UPDATE sessions SET name = ?, last_modified = ?, vector_store_path = ? WHERE id = ?",
            (session_name, now, vector_store_path, session_id)
        )
    
    # Clear existing document associations
    cursor.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
    
    # Add document associations
    for doc_id in document_ids:
        cursor.execute(
            "INSERT INTO session_documents (session_id, document_id) VALUES (?, ?)",
            (session_id, doc_id)
        )
    
    # Save messages
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    for msg in st.session_state.messages:
        msg_id = hashlib.md5(f"{session_id}_{msg['role']}_{msg['content']}_{datetime.now().isoformat()}".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (msg_id, session_id, msg['role'], msg['content'], now)
        )
    
    conn.commit()
    conn.close()
    
    # Update session metadata in session state
    st.session_state.sessions_metadata[session_name] = {
        "id": session_id,
        "created": now,
        "last_modified": now,
        "document_count": len(document_ids),
        "message_count": len(st.session_state.messages)
    }
    
    return session_id


# Modify the load_session_from_db function to add the allow_dangerous_deserialization parameter
def load_session_from_db(session_name):
    """Load a session from the database"""
    if session_name not in st.session_state.sessions_metadata:
        st.error(f"Session '{session_name}' not found.")
        return False

    session_id = st.session_state.sessions_metadata[session_name]["id"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get session info
    cursor.execute("SELECT vector_store_path FROM sessions WHERE id = ?", (session_id,))
    session_info = cursor.fetchone()
    if not session_info:
        conn.close()
        st.error(f"Session data for '{session_name}' not found in database.")
        return False

    vector_store_path = session_info[0]

    # Get documents for this session
    cursor.execute(
        "SELECT d.id, d.name, d.pages FROM documents d " +
        "JOIN session_documents sd ON d.id = sd.document_id " +
        "WHERE sd.session_id = ?",
        (session_id,)
    )
    documents = cursor.fetchall()

    # Get messages for this session
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
    messages = cursor.fetchall()

    conn.close()

    # Update session state
    st.session_state.session_id = session_id
    st.session_state.processed_files = [{"id": doc[0], "name": doc[1], "pages": doc[2]} for doc in documents]
    st.session_state.messages = [{"role": msg[0], "content": msg[1]} for msg in messages]

    # Get document IDs for this session
    document_ids = [doc[0] for doc in documents]

    # Try to load vector store from database first
    try:
        # Get the model name used for embeddings (defaulting to "openai")
        model_name = "openai"

        # Create vector store from stored embeddings
        db_vector_store = create_vector_store_from_db(document_ids, model_name)

        if db_vector_store:
            st.session_state.vector_store = db_vector_store
            st.session_state.conversation = get_conversation_chain(db_vector_store)
            st.info("Successfully loaded embeddings from database.")
            return True
    except Exception as e:
        st.warning(f"Could not load embeddings from database: {str(e)}")
        # Fall back to file-based vector store

    # Fall back to loading vector store from file if it exists
    # Fall back to loading vector store from file if it exists
    if vector_store_path and os.path.exists(vector_store_path):
        try:
            embeddings = OpenAIEmbeddings()
            st.session_state.vector_store = FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            # Recreate conversation chain
            st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False

    return True


# Also modify the save_session_to_db function to ensure consistent behavior when saving
def save_session_to_db(session_name, document_ids, vector_store_path=None):
    """Save a session to the database"""
    # Generate session ID if needed
    if not st.session_state.session_id:
        st.session_state.session_id = hashlib.md5(f"{session_name}_{datetime.now().isoformat()}".encode()).hexdigest()

    session_id = st.session_state.session_id
    now = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if session exists
    cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if cursor.fetchone() is None:
        # Create new session
        cursor.execute(
            "INSERT INTO sessions (id, name, created, last_modified, vector_store_path) VALUES (?, ?, ?, ?, ?)",
            (session_id, session_name, now, now, vector_store_path)
        )
    else:
        # Update existing session
        cursor.execute(
            "UPDATE sessions SET name = ?, last_modified = ?, vector_store_path = ? WHERE id = ?",
            (session_name, now, vector_store_path, session_id)
        )

    # Clear existing document associations
    cursor.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))

    # Add document associations
    for doc_id in document_ids:
        cursor.execute(
            "INSERT INTO session_documents (session_id, document_id) VALUES (?, ?)",
            (session_id, doc_id)
        )

    # Save messages
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    for msg in st.session_state.messages:
        msg_id = hashlib.md5(
            f"{session_id}_{msg['role']}_{msg['content']}_{datetime.now().isoformat()}".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (msg_id, session_id, msg['role'], msg['content'], now)
        )

    conn.commit()
    conn.close()

    # Update session metadata in session state
    st.session_state.sessions_metadata[session_name] = {
        "id": session_id,
        "created": now,
        "last_modified": now,
        "document_count": len(document_ids),
        "message_count": len(st.session_state.messages)
    }

    return session_id

def delete_session_from_db(session_name):
    """Delete a session from the database"""
    if session_name not in st.session_state.sessions_metadata:
        st.error(f"Session '{session_name}' not found.")
        return False
    
    session_id = st.session_state.sessions_metadata[session_name]["id"]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get vector store path to delete files
    cursor.execute("SELECT vector_store_path FROM sessions WHERE id = ?", (session_id,))
    vector_store_path = cursor.fetchone()[0]
    
    # Delete related records
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    
    conn.commit()
    conn.close()
    
    # Delete vector store files if they exist
    if vector_store_path and os.path.exists(vector_store_path):
        import shutil
        shutil.rmtree(vector_store_path)
    
    # Remove from session state
    del st.session_state.sessions_metadata[session_name]
    
    return True

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files"""
    processed_files = []
    all_text = ""
    
    for pdf in pdf_files:
        # Save PDF to database
        doc_info = save_pdf_to_db(pdf, pdf.name)
        
        if doc_info:
            # Extract text from the saved document
            doc_text = extract_text_from_db_document(doc_info["id"])
            all_text += doc_text
            
            # Add to processed files list
            processed_files.append(doc_info)
    
    return all_text, processed_files

def auto_handle_embeddings(uploaded_files=None, selected_doc_ids=None):
    """Automatically handle embeddings for uploaded files and selected documents

    This function:
    1. Processes newly uploaded files
    2. Extracts text and creates chunks
    3. Checks if embeddings exist in the database
    4. Uses stored embeddings when available
    5. Creates new embeddings when needed and stores them
    6. Combines everything into a unified vector store
    """
    with st.spinner("Processing documents and managing embeddings..."):
        text_to_process = ""
        newly_processed_files = []
        all_doc_ids = []

        # Process uploaded files if any
        if uploaded_files:
            extracted_text, new_processed_files = extract_text_from_pdfs(uploaded_files)
            text_to_process += extracted_text
            newly_processed_files.extend(new_processed_files)
            all_doc_ids.extend([doc["id"] for doc in new_processed_files])

        # Add selected document IDs if any
        if selected_doc_ids:
            all_doc_ids.extend([doc_id for doc_id in selected_doc_ids
                                if doc_id not in [doc["id"] for doc in newly_processed_files]])

        # Exit if no documents to process
        if not all_doc_ids:
            st.warning("No documents selected or uploaded for processing.")
            return False

        # Update processed files list
        current_processed_ids = [doc.get("id") for doc in st.session_state.processed_files]
        for new_doc in newly_processed_files:
            if new_doc.get("id") not in current_processed_ids:
                st.session_state.processed_files.append(new_doc)

        # For selected documents that aren't newly uploaded, add them to processed files
        if selected_doc_ids:
            for doc_id in selected_doc_ids:
                if doc_id not in current_processed_ids and doc_id not in [doc["id"] for doc in newly_processed_files]:
                    doc = get_document_from_db(doc_id)
                    if doc:
                        st.session_state.processed_files.append({
                            "id": doc_id,
                            "name": doc["name"],
                            "pages": doc["pages"]
                        })

        # Create a dictionary to track which documents need new embeddings
        docs_needing_embeddings = {}

        # Check which documents already have embeddings in the database
        have_embeddings = []
        need_embeddings = []

        for doc_id in all_doc_ids:
            # Get chunks for this document
            chunks = get_chunks_for_document(doc_id)

            if chunks:
                # Get chunk IDs
                chunk_ids = [chunk["id"] for chunk in chunks]

                # Get embeddings for these chunks
                model_name = "openai"  # Default model name
                embeddings_dict = get_embeddings_for_chunks(chunk_ids, model_name)

                if embeddings_dict and len(embeddings_dict) == len(chunk_ids):
                    # Document has all embeddings
                    have_embeddings.append(doc_id)
                else:
                    # Document needs new embeddings
                    need_embeddings.append(doc_id)

                    # Extract text if this is a selected document (not newly uploaded)
                    if doc_id not in [doc["id"] for doc in newly_processed_files]:
                        doc_text = extract_text_from_db_document(doc_id)
                        if doc_text:
                            text_to_process += doc_text
                            docs_needing_embeddings[doc_id] = doc_text
            else:
                # No chunks found, document needs processing
                need_embeddings.append(doc_id)

                # Extract text if this is a selected document (not newly uploaded)
                if doc_id not in [doc["id"] for doc in newly_processed_files]:
                    doc_text = extract_text_from_db_document(doc_id)
                    if doc_text:
                        text_to_process += doc_text
                        docs_needing_embeddings[doc_id] = doc_text

        # Create final vector store
        final_vector_store = None

        # 1. First try to load existing embeddings from database
        if have_embeddings:
            st.info(f"Loading existing embeddings for {len(have_embeddings)} document(s)...")
            try:
                db_vector_store = create_vector_store_from_db(have_embeddings)
                if db_vector_store:
                    final_vector_store = db_vector_store
                    st.success(f"Successfully loaded embeddings for {len(have_embeddings)} document(s)!")
            except Exception as e:
                st.warning(f"Error loading embeddings from database: {str(e)}")

        # 2. Process documents that need new embeddings
        if need_embeddings and text_to_process:
            st.info(f"Creating new embeddings for {len(need_embeddings)} document(s)...")

            try:
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                all_chunks = text_splitter.split_text(text_to_process)

                # Create embeddings model with the hardcoded API key
                embeddings_model = OpenAIEmbeddings()
                
                # Get model name
                model_name = "openai"
                if hasattr(embeddings_model, "model"):
                    model_name = embeddings_model.model
                elif hasattr(embeddings_model, "model_name"):
                    model_name = embeddings_model.model_name

                # Create embeddings for all chunks
                embedding_vectors = embeddings_model.embed_documents(all_chunks)

                # Create vector store with new embeddings
                new_vector_store = FAISS.from_texts(texts=all_chunks, embedding=embeddings_model)

                # Save chunks and embeddings to database
                # For newly uploaded files, we can use the entire set of chunks
                for doc_id in [doc["id"] for doc in newly_processed_files]:
                    if doc_id in need_embeddings:
                        save_chunks_to_db(doc_id, all_chunks, embedding_vectors, model_name)

                # For other documents, we need to split their text separately
                for doc_id, doc_text in docs_needing_embeddings.items():
                    if doc_id not in [doc["id"] for doc in newly_processed_files]:
                        doc_chunks = text_splitter.split_text(doc_text)
                        doc_embeddings = embeddings_model.embed_documents(doc_chunks)
                        save_chunks_to_db(doc_id, doc_chunks, doc_embeddings, model_name)

                # Merge with existing vector store or set as the final one
                if final_vector_store:
                    final_vector_store.merge_from(new_vector_store)
                else:
                    final_vector_store = new_vector_store

                st.success(f"Successfully created and saved embeddings for {len(need_embeddings)} document(s)!")

            except Exception as e:
                st.error(f"Error creating new embeddings: {str(e)}")
                return False

        # Update session state with the final vector store
        if final_vector_store:
            st.session_state.vector_store = final_vector_store
            st.session_state.conversation = get_conversation_chain(final_vector_store)
            return True
        else:
            st.warning("Could not create or retrieve embeddings for the documents.")
            return False

def auto_process_uploaded_files(uploaded_files):
    """Process newly uploaded files automatically with automatic embedding management"""
    if not uploaded_files:
        return

    # Compare with previous uploads to find new files
    current_filenames = [file.name for file in uploaded_files]
    previous_filenames = st.session_state.previous_uploaded_files

    # Find new files
    new_files = []
    for i, file in enumerate(uploaded_files):
        if file.name not in previous_filenames:
            new_files.append(file)

    # Update the list of previous uploads
    st.session_state.previous_uploaded_files = current_filenames

    # Process new files if any
    if new_files:
        st.info(f"Auto-processing {len(new_files)} new file(s)...")
        auto_handle_embeddings(uploaded_files=new_files)

def process_documents(uploaded_files=None, selected_doc_ids=None):
    """Process documents using automatic embedding handling"""
    return auto_handle_embeddings(uploaded_files, selected_doc_ids)

def create_vector_store_from_db(doc_ids, model_name="openai"):
    """Create a vector store from chunks and embeddings stored in the database"""
    if not doc_ids:
        return None

    # Collect all chunks from the specified documents
    all_chunks = []
    all_chunk_ids = []

    for doc_id in doc_ids:
        chunks = get_chunks_for_document(doc_id)
        all_chunks.extend([chunk["content"] for chunk in chunks])
        all_chunk_ids.extend([chunk["id"] for chunk in chunks])

    if not all_chunks:
        return None

    # Check if we have embeddings for these chunks
    embeddings_dict = get_embeddings_for_chunks(all_chunk_ids, model_name)

    # If we have embeddings for all chunks, use them to create the vector store
    if embeddings_dict and len(embeddings_dict) == len(all_chunk_ids):
        try:
            # Get embeddings in the same order as chunks
            embedding_vectors = [embeddings_dict[chunk_id] for chunk_id in all_chunk_ids]

            # Create embeddings model with the hardcoded API key
            embeddings_model = OpenAIEmbeddings()

            # Create FAISS index from existing vectors
            import numpy as np
            from faiss import IndexFlatL2

            # Convert to numpy array
            vectors = np.array(embedding_vectors).astype('float32')

            # Create index
            index = IndexFlatL2(vectors.shape[1])
            index.add(vectors)

            # Create FAISS vector store with pre-computed embeddings
            vector_store = FAISS(embeddings_model.embed_query, index, all_chunks, {})

            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store from stored embeddings: {str(e)}")
            # Fall back to creating new embeddings

    # If we don't have embeddings or there was an error, return None
    # The calling function should handle creating new embeddings
    return None

def get_conversation_chain(vector_store):
    """Create a conversation chain using the vector store"""
    try:
        # Use gpt-4o-mini as the fixed model
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
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

        # Create conversation chain with the retriever from our vector store
        # The vector store already uses our consistent embedding model
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

# Initialize the database
init_database()

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
    st.session_state.sessions_metadata = {}
    # Load sessions from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, created, last_modified FROM sessions")
    sessions = cursor.fetchall()
    
    for session in sessions:
        session_id, name, created, last_modified = session
        # Count documents for this session
        cursor.execute("SELECT COUNT(*) FROM session_documents WHERE session_id = ?", (session_id,))
        doc_count = cursor.fetchone()[0]
        
        # Count messages for this session
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        msg_count = cursor.fetchone()[0]
        
        st.session_state.sessions_metadata[name] = {
            "id": session_id,
            "created": created,
            "last_modified": last_modified,
            "document_count": doc_count,
            "message_count": msg_count
        }
    
    conn.close()

# Keep track of previously uploaded files to detect new ones
if "previous_uploaded_files" not in st.session_state:
    st.session_state.previous_uploaded_files = []

# Auto-load most recent session if none is active
if not st.session_state.session_id and st.session_state.sessions_metadata:
    # Find the most recently modified session
    most_recent_session = None
    most_recent_timestamp = None

    for name, metadata in st.session_state.sessions_metadata.items():
        timestamp = metadata.get("last_modified")
        if timestamp and (most_recent_timestamp is None or timestamp > most_recent_timestamp):
            most_recent_timestamp = timestamp
            most_recent_session = name

    # Load the most recent session
    if most_recent_session:
        if load_session_from_db(most_recent_session):
            st.success(f"Automatically loaded your most recent session: '{most_recent_session}'")

# Initialize the embedding model with the hardcoded API key
try:
    st.session_state.embedding_model = OpenAIEmbeddings()
except Exception as e:
    st.error(f"Error initializing embedding model: {str(e)}")
    st.session_state.embedding_model = None

# Create layout with two columns
col1, col2 = st.columns([2, 1])

# Main content in first column
with col1:
    # Display chat messages
    st.subheader("Chat with your documents")
    
    # Show chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are processed
        if st.session_state.conversation is None:
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
                            # Get document IDs for this session
                            doc_ids = [doc["id"] for doc in st.session_state.processed_files]
                            # Create vector store path
                            vector_store_path = os.path.join(SAVE_DIR, st.session_state.session_id, "vector_store")
                            # Save session to database
                            save_session_to_db(session_name, doc_ids, vector_store_path)
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar and configuration in second column
with col2:
    # Configuration section
    st.subheader("Configuration")
    st.info("Using gpt-4o-mini model")
    
    # Upload Documents section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Auto-process newly uploaded files
    if uploaded_files:
        auto_process_uploaded_files(uploaded_files)
    
    # Session Management section
    st.subheader("Session Management")
    
    # Save current session
    session_name = st.text_input("Session Name", placeholder="Enter a name for this session")
    save_button = st.button("Save Current Session")
    
    if save_button and session_name:
        # Save vector store if it exists
        vector_store_path = None
        if st.session_state.vector_store:
            vector_store_path = os.path.join(SAVE_DIR, st.session_state.session_id or hashlib.md5(session_name.encode()).hexdigest(), "vector_store")
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            st.session_state.vector_store.save_local(vector_store_path)
        
        # Get document IDs for this session
        doc_ids = [doc["id"] for doc in st.session_state.processed_files]
        
        # Save session to database
        save_session_to_db(session_name, doc_ids, vector_store_path)
        
        st.success(f"Session '{session_name}' saved successfully!")
    
    # Load existing session
    st.subheader("Load Existing Session")
    session_options = list(st.session_state.sessions_metadata.keys())
    
    if session_options:
        selected_session = st.selectbox("Select a session to load", session_options)
        load_button = st.button("Load Selected Session")
        
        if load_button and selected_session:
            if load_session_from_db(selected_session):
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
            if delete_session_from_db(delete_session):
                st.success(f"Session '{delete_session}' deleted successfully!")
                st.rerun()

# Show which documents are currently active in the session
if st.session_state.processed_files:
    st.sidebar.subheader("Active Documents in Session")
    for doc in st.session_state.processed_files:
        st.sidebar.write(f"📄 {doc.get('name')} - {doc.get('pages')} pages")

# Footer with additional information
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown("""
This RAG chatbot uses a database to store and manage PDF documents:

- PDFs are stored in SQLite for persistence
- Text chunks and their vector embeddings are stored in the database
- Embeddings are reused when possible to reduce API costs
- Performance improves with each session as more embeddings are cached
- Sessions track document references rather than duplicating content
""")

# Add embedding stats
st.sidebar.markdown("---")
st.sidebar.subheader("Embedding Statistics")

# Count embeddings in database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM embeddings")
embedding_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(DISTINCT model) FROM embeddings")
model_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM chunks")
chunk_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM documents")
doc_count = cursor.fetchone()[0]
conn.close()

st.sidebar.markdown(f"""
- Total documents: {doc_count}
- Total text chunks: {chunk_count}
- Stored embeddings: {embedding_count}
- Embedding models used: {model_count}
""")