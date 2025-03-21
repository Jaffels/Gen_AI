import streamlit as st
import os
import json
import hashlib
import io
import base64
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import (ConversationBufferMemory)
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import tempfile
import pandas as pd
import sqlitecloud

# App title and configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
st.title("RAG Chatbot with SQLiteCloud")

# Create directory for saved sessions if it doesn't exist
SAVE_DIR = os.path.join(os.getcwd(), "saved_sessions")
os.makedirs(SAVE_DIR, exist_ok=True)

# SQLiteCloud adapter class
class SQLiteCloudAdapter:
    """Adapter for SQLiteCloud database operations"""
    
    def __init__(self, host="", port=4444, key="", username="", password="", database="rag_chatbot", local_fallback=True):
        self.host = host
        self.port = port
        self.key = key
        self.username = username
        self.password = password
        self.database = database
        self.local_fallback = local_fallback
        self.local_db_path = os.path.join(os.getcwd(), "rag_chatbot.db")
        self.cloud_enabled = False
        
    def build_connection_string(self):
        """Build SQLiteCloud connection string"""
        # Format: sqlitecloud://host:port/database?apikey=key
        # or sqlitecloud://username:password@host:port/database?key=key
        if self.username and self.password:
            return f"sqlitecloud://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?key={self.key}"
        else:
            return f"sqlitecloud://{self.host}:{self.port}/{self.database}?apikey={self.key}"
    
    def connect(self):
        """Connect to SQLiteCloud or local SQLite based on configuration"""
        # Use the direct SQLiteCloud connection string
        if self.cloud_enabled:
            try:
                # Using the exact connection string provided
                conn = sqlitecloud.connect("sqlitecloud://cd195vehhz.g6.sqlite.cloud:8860/chinook.sqlite?apikey=cZ77GLQCoIWLbzgMwZPXTTDq8rYARAc1PnaWxn4lb8Q")
                self.cloud_enabled = True
                return conn
            except Exception as e:
                st.sidebar.error(f"SQLiteCloud connection error: {str(e)}")
                if not self.local_fallback:
                    return None
                st.sidebar.warning("Falling back to local SQLite database")
        
        # Fall back to local SQLite
        self.cloud_enabled = False
        try:
            import sqlite3
            return sqlite3.connect(self.local_db_path)
        except Exception as e:
            st.sidebar.error(f"Local database connection error: {str(e)}")
            return None
    
    def execute_query(self, query, params=(), fetch=None, commit=True):
        """Execute SQL query with appropriate connection"""
        conn = self.connect()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if commit and not self.cloud_enabled:
                conn.commit()
                
            if fetch == 'one':
                result = cursor.fetchone()
            elif fetch == 'all':
                result = cursor.fetchall()
            else:
                result = True
                
            # SQLiteCloud manages its own connections, only close for local SQLite
            if not self.cloud_enabled:
                conn.close()
                
            return result
        except Exception as e:
            st.sidebar.error(f"Query execution error: {str(e)}")
            if not self.cloud_enabled:
                conn.close()
            return None
    
    def execute_many(self, query, params_list):
        """Execute many queries at once"""
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            
            if not self.cloud_enabled:
                conn.commit()
                conn.close()
                
            return True
        except Exception as e:
            st.sidebar.error(f"Execute many error: {str(e)}")
            if not self.cloud_enabled:
                conn.close()
            return False
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        # Create table for PDF documents
        self.execute_query('''
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
        self.execute_query('''
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        ''')

        # Create table for embeddings
        self.execute_query('''
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
        self.execute_query('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created TEXT NOT NULL,
            last_modified TEXT NOT NULL,
            vector_store_path TEXT
        )
        ''')

        # Create table for session_documents (many-to-many relationship)
        self.execute_query('''
        CREATE TABLE IF NOT EXISTS session_documents (
            session_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            PRIMARY KEY (session_id, document_id),
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        ''')

        # Create table for messages
        self.execute_query('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        ''')


# SQLiteCloud configuration UI
def setup_sqlitecloud():
    """Setup SQLiteCloud configuration in the sidebar"""
    st.sidebar.subheader("SQLiteCloud Configuration")
    
    # Toggle to enable/disable SQLiteCloud
    use_cloud = st.sidebar.toggle("Use SQLiteCloud Database", 
                                  value=st.session_state.get("use_sqlitecloud", False))
    st.session_state["use_sqlitecloud"] = use_cloud
    
    # Simple message when cloud is enabled
    if use_cloud:
        st.sidebar.success("Using SQLiteCloud database: chinook.sqlite")
        
        # Test connection button
        if st.sidebar.button("Test Connection"):
            try:
                conn = sqlitecloud.connect("sqlitecloud://cd195vehhz.g6.sqlite.cloud:8860/chinook.sqlite?apikey=cZ77GLQCoIWLbzgMwZPXTTDq8rYARAc1PnaWxn4lb8Q")
                st.sidebar.success("Successfully connected to SQLiteCloud!")
                st.session_state["cloud_connected"] = True
                conn.close()  # Close the connection after successful test
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
                st.session_state["cloud_connected"] = False
    
    # Create adapter with hard-coded connection
    adapter = SQLiteCloudAdapter(
        host="cd195vehhz.g6.sqlite.cloud",
        port=8860,
        key="cZ77GLQCoIWLbzgMwZPXTTDq8rYARAc1PnaWxn4lb8Q",
        database="chinook.sqlite",
        local_fallback=True
    )
    
    # Set cloud_enabled based on user toggle
    adapter.cloud_enabled = use_cloud
    
    # Initialize database schema
    adapter.init_database()
    
    return adapter

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

# Initialize the database adapter
db_adapter = setup_sqlitecloud()

# Load sessions from database
def load_sessions_metadata():
    """Load sessions metadata from database"""
    sessions = db_adapter.execute_query(
        "SELECT id, name, created, last_modified FROM sessions",
        fetch='all'
    ) or []
    
    st.session_state.sessions_metadata = {}
    for session in sessions:
        session_id, name, created, last_modified = session
        
        # Count documents for this session
        doc_count = db_adapter.execute_query(
            "SELECT COUNT(*) FROM session_documents WHERE session_id = ?",
            (session_id,),
            fetch='one'
        )
        doc_count = doc_count[0] if doc_count else 0

        # Count messages for this session
        msg_count = db_adapter.execute_query(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
            fetch='one'
        )
        msg_count = msg_count[0] if msg_count else 0

        st.session_state.sessions_metadata[name] = {
            "id": session_id,
            "created": created,
            "last_modified": last_modified,
            "document_count": doc_count,
            "message_count": msg_count
        }

# Load sessions metadata
load_sessions_metadata()

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

    # Check if document already exists
    existing_doc = db_adapter.execute_query(
        "SELECT id FROM documents WHERE id = ?",
        (doc_id,),
        fetch='one'
    )
    
    if not existing_doc:
        # Insert new document
        db_adapter.execute_query(
            "INSERT INTO documents (id, name, content, pages, upload_date, file_size, processed) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, filename, pdf_content, page_count, datetime.now().isoformat(), file_size, False)
        )

    return {
        "id": doc_id,
        "name": filename,
        "pages": page_count,
        "size": file_size,
        "upload_date": datetime.now().isoformat()
    }


def get_document_from_db(doc_id):
    """Retrieve a document from the database by ID"""
    doc = db_adapter.execute_query(
        "SELECT id, name, content, pages, upload_date, file_size, processed FROM documents WHERE id = ?",
        (doc_id,),
        fetch='one'
    )

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
    docs = db_adapter.execute_query(
        "SELECT id, name, pages, upload_date, file_size, processed FROM documents",
        fetch='all'
    ) or []

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
        db_adapter.execute_query(
            "UPDATE documents SET processed = TRUE WHERE id = ?",
            (doc_id,)
        )

        return text
    except Exception as e:
        st.error(f"Error extracting text from {doc['name']}: {str(e)}")
        return ""
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def save_chunks_to_db(doc_id, chunks, embeddings_list=None, model_name=None):
    """Save text chunks to the database with optional embeddings"""
    # First delete any existing chunks and their embeddings for this document
    # Get all chunk ids for this document
    chunk_ids = db_adapter.execute_query(
        "SELECT id FROM chunks WHERE document_id = ?",
        (doc_id,),
        fetch='all'
    )
    chunk_ids = [row[0] for row in chunk_ids] if chunk_ids else []

    # Delete associated embeddings
    if chunk_ids:
        placeholders = ','.join(['?'] * len(chunk_ids))
        db_adapter.execute_query(
            f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )

    # Delete the chunks
    db_adapter.execute_query(
        "DELETE FROM chunks WHERE document_id = ?",
        (doc_id,)
    )

    # Insert new chunks
    chunk_ids = []
    chunk_params = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{doc_id}_{i}_{chunk}".encode()).hexdigest()
        chunk_ids.append(chunk_id)
        chunk_params.append((chunk_id, doc_id, chunk, i))
    
    # Use executemany for better performance with multiple inserts
    db_adapter.execute_many(
        "INSERT INTO chunks (id, document_id, content, chunk_index) VALUES (?, ?, ?, ?)",
        chunk_params
    )

    # Insert embeddings if provided
    if embeddings_list and model_name and len(embeddings_list) == len(chunks):
        embedding_params = []
        for chunk_id, embedding_vector in zip(chunk_ids, embeddings_list):
            embedding_id = hashlib.md5(f"{chunk_id}_{model_name}".encode()).hexdigest()
            # Serialize the embedding vector to binary
            embedding_blob = base64.b64encode(json.dumps(embedding_vector).encode())
            embedding_params.append((
                embedding_id, 
                chunk_id, 
                embedding_blob, 
                model_name, 
                datetime.now().isoformat()
            ))
        
        db_adapter.execute_many(
            "INSERT INTO embeddings (id, chunk_id, embedding, model, created_at) VALUES (?, ?, ?, ?, ?)",
            embedding_params
        )


def get_chunks_for_document(doc_id):
    """Get all chunks for a document"""
    chunks = db_adapter.execute_query(
        "SELECT id, content, chunk_index FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc_id,),
        fetch='all'
    ) or []

    return [{"id": chunk[0], "content": chunk[1], "index": chunk[2]} for chunk in chunks]


def get_embeddings_for_chunks(chunk_ids, model_name=None):
    """Get embeddings for specific chunks, optionally filtered by model"""
    if not chunk_ids:
        return {}
    
    placeholders = ','.join(['?'] * len(chunk_ids))
    
    if model_name:
        query = f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders}) AND model = ?"
        results = db_adapter.execute_query(
            query,
            chunk_ids + [model_name],
            fetch='all'
        ) or []
    else:
        query = f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})"
        results = db_adapter.execute_query(
            query,
            chunk_ids,
            fetch='all'
        ) or []

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

    # Check if session exists
    existing_session = db_adapter.execute_query(
        "SELECT id FROM sessions WHERE id = ?",
        (session_id,),
        fetch='one'
    )
    
    if not existing_session:
        # Create new session
        db_adapter.execute_query(
            "INSERT INTO sessions (id, name, created, last_modified, vector_store_path) VALUES (?, ?, ?, ?, ?)",
            (session_id, session_name, now, now, vector_store_path)
        )
    else:
        # Update existing session
        db_adapter.execute_query(
            "UPDATE sessions SET name = ?, last_modified = ?, vector_store_path = ? WHERE id = ?",
            (session_name, now, vector_store_path, session_id)
        )

    # Clear existing document associations
    db_adapter.execute_query(
        "DELETE FROM session_documents WHERE session_id = ?",
        (session_id,)
    )

    # Add document associations
    doc_params = [(session_id, doc_id) for doc_id in document_ids]
    db_adapter.execute_many(
        "INSERT INTO session_documents (session_id, document_id) VALUES (?, ?)",
        doc_params
    )

    # Save messages
    db_adapter.execute_query(
        "DELETE FROM messages WHERE session_id = ?",
        (session_id,)
    )
    
    msg_params = []
    for msg in st.session_state.messages:
        msg_id = hashlib.md5(
            f"{session_id}_{msg['role']}_{msg['content']}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        msg_params.append((
            msg_id, 
            session_id, 
            msg['role'], 
            msg['content'], 
            now
        ))
    
    db_adapter.execute_many(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        msg_params
    )

    # Update session metadata in session state
    st.session_state.sessions_metadata[session_name] = {
        "id": session_id,
        "created": now,
        "last_modified": now,
        "document_count": len(document_ids),
        "message_count": len(st.session_state.messages)
    }

    return session_id


def load_session_from_db(session_name):
    """Load a session from the database"""
    if session_name not in st.session_state.sessions_metadata:
        st.error(f"Session '{session_name}' not found.")
        return False

    session_id = st.session_state.sessions_metadata[session_name]["id"]

    # Get session info
    session_info = db_adapter.execute_query(
        "SELECT vector_store_path FROM sessions WHERE id = ?",
        (session_id,),
        fetch='one'
    )
    
    if not session_info:
        st.error(f"Session data for '{session_name}' not found in database.")
        return False

    vector_store_path = session_info[0]

    # Get documents for this session
    documents = db_adapter.execute_query(
        "SELECT d.id, d.name, d.pages FROM documents d " +
        "JOIN session_documents sd ON d.id = sd.document_id " +
        "WHERE sd.session_id = ?",
        (session_id,),
        fetch='all'
    ) or []

    # Get messages for this session
    messages = db_adapter.execute_query(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
        fetch='all'
    ) or []

    # Get document IDs for this session
    document_ids = db_adapter.execute_query(
        "SELECT document_id FROM session_documents WHERE session_id = ?",
        (session_id,),
        fetch='all'
    ) or []
    document_ids = [row[0] for row in document_ids]

    # Update session state
    st.session_state.session_id = session_id
    st.session_state.processed_files = [{"id": doc[0], "name": doc[1], "pages": doc[2]} for doc in documents]
    st.session_state.messages = [{"role": msg[0], "content": msg[1]} for msg in messages]

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
    if vector_store_path and os.path.exists(vector_store_path):
        try:
            embeddings = OpenAIEmbeddings()
            st.session_state.vector_store = FAISS.load_local(vector_store_path, embeddings)

            # Recreate conversation chain
            st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False

    return True


def delete_session_from_db(session_name):
    """Delete a session from the database"""
    if session_name not in st.session_state.sessions_metadata:
        st.error(f"Session '{session_name}' not found.")
        return False

    session_id = st.session_state.sessions_metadata[session_name]["id"]

    # Get vector store path to delete files
    vector_store_path = db_adapter.execute_query(
        "SELECT vector_store_path FROM sessions WHERE id = ?",
        (session_id,),
        fetch='one'
    )
    vector_store_path = vector_store_path[0] if vector_store_path else None

    # Delete related records
    db_adapter.execute_query(
        "DELETE FROM messages WHERE session_id = ?",
        (session_id,)
    )
    
    db_adapter.execute_query(
        "DELETE FROM session_documents WHERE session_id = ?",
        (session_id,)
    )
    
    db_adapter.execute_query(
        "DELETE FROM sessions WHERE id = ?",
        (session_id,)
    )

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
            
            st.info(f"Processed {doc_info['name']} ({doc_info['pages']} pages)")

    return all_text, processed_files


# Function to create vector store from text
def create_vector_store(text):
    """Create a vector store from text and store embeddings in the database"""
    if not text:
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    if not st.session_state.get("api_key"):
        st.warning("Please enter your OpenAI API key to enable embeddings.")
        return None

    try:
        embeddings_model = OpenAIEmbeddings()

        # Create the vector store
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings_model)

        # Extract the embedding vectors
        # This is a bit hacky since we need to access the private _embeddings attribute
        # In a production system, you'd want to use the proper API or compute embeddings separately
        embedding_vectors = None
        if hasattr(vector_store, "_embeddings"):
            embedding_vectors = vector_store._embeddings

        # Get the model name
        model_name = "openai"
        if hasattr(embeddings_model, "model"):
            model_name = embeddings_model.model

        # Save chunks and embeddings to database for documents in this session
        for doc in st.session_state.processed_files:
            doc_id = doc.get("id")
            if doc_id:
                save_chunks_to_db(doc_id, chunks, embedding_vectors, model_name)

        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None


# Function to process selected documents from database - kept but no longer used directly
def process_selected_documents(doc_ids):
    """Process selected documents from the database"""
    all_text = ""
    processed_files = []

    for doc_id in doc_ids:
        # Get document info
        doc = get_document_from_db(doc_id)
        if doc:
            # Extract text
            doc_text = extract_text_from_db_document(doc_id)
            all_text += doc_text

            # Add to processed files
            processed_files.append({
                "id": doc_id,
                "name": doc["name"],
                "pages": doc["pages"]
            })

    return all_text, processed_files


# Function to create conversation chain
def get_conversation_chain(vector_store):
    """Create a conversation chain using the vector store"""
    if not st.session_state.get("api_key"):
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

            # Create embedding model
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

    # If we don't have embeddings or there was an error, create new ones
    try:
        embeddings_model = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts=all_chunks, embedding=embeddings_model)

        # Store the new embeddings
        if hasattr(vector_store, "_embeddings"):
            embedding_vectors = vector_store._embeddings

            # Save embeddings for each document's chunks
            chunk_index = 0
            for doc_id in doc_ids:
                chunks = get_chunks_for_document(doc_id)
                doc_chunks = []
                doc_embeddings = []

                for i in range(len(chunks)):
                    doc_chunks.append(chunks[i]["content"])
                    doc_embeddings.append(embedding_vectors[chunk_index])
                    chunk_index += 1

                save_chunks_to_db(doc_id, doc_chunks, doc_embeddings, model_name)

        return vector_store
    except Exception as e:
        st.error(f"Error creating new embeddings: {str(e)}")
        return None


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
        # Check if OpenAI API key is provided
        if not st.session_state.get("api_key"):
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
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    model_option = st.selectbox(
        "Select OpenAI Model",
        ("gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4o")
    )

    # Upload Documents section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # Process documents automatically when files are uploaded
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            text_to_process = ""
            newly_processed_files = []
            processed_doc_ids = []

            # Process uploaded files
            extracted_text, new_processed_files = extract_text_from_pdfs(uploaded_files)
            text_to_process += extracted_text
            newly_processed_files.extend(new_processed_files)
            processed_doc_ids.extend([doc["id"] for doc in new_processed_files])

            if text_to_process:
                # Update processed files list
                st.session_state.processed_files = newly_processed_files

                # Create vector store
                new_vector_store = create_vector_store(text_to_process)

                if new_vector_store:
                    st.session_state.vector_store = new_vector_store

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(new_vector_store)
                    st.success(f"Successfully processed {len(newly_processed_files)} document(s)!")
                else:
                    # Try to create vector store from database
                    st.info("Attempting to retrieve embeddings from database...")
                    db_vector_store = create_vector_store_from_db(processed_doc_ids)

                    if db_vector_store:
                        st.session_state.vector_store = db_vector_store
                        st.session_state.conversation = get_conversation_chain(db_vector_store)
                        st.success(
                            f"Successfully loaded embeddings from database for {len(newly_processed_files)} document(s)!")
                    else:
                        st.warning("Could not create vector store from the documents.")

    # Show stored documents
    st.subheader("Document Library")

    # Fetch all documents from database
    all_docs = get_all_documents()

    if all_docs:
        # Create a dataframe for better display
        doc_df = pd.DataFrame([
            {
                "Name": doc["name"],
                "Pages": doc["pages"],
                "Date": doc["upload_date"].split("T")[0],
                "Size (KB)": round(doc["size"] / 1024, 1),
                "ID": doc["id"]
            }
            for doc in all_docs
        ])

        # Display as table without selection checkboxes
        st.dataframe(
            doc_df,
            column_config={
                "ID": st.column_config.Column(
                    "ID",
                    disabled=True
                )
            },
            hide_index=True
        )
    else:
        st.info("No documents in the database. Upload some PDFs to get started.")

    # Session Management section
    st.subheader("Session Management")

    # Save current session
    session_name = st.text_input("Session Name", placeholder="Enter a name for this session")
    save_button = st.button("Save Current Session")

    if save_button and session_name:
        # Save vector store if it exists
        vector_store_path = None
        if st.session_state.vector_store:
            vector_store_path = os.path.join(SAVE_DIR, st.session_state.session_id or hashlib.md5(
                session_name.encode()).hexdigest(), "vector_store")
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
This RAG chatbot uses SQLiteCloud to store and manage PDF documents:

- PDFs are stored in SQLiteCloud for persistence across sessions
- Text chunks and their vector embeddings are stored in the database
- Embeddings are reused when possible to reduce API costs
- Performance improves with each session as more embeddings are cached
- Sessions track document references rather than duplicating content

Connection:
```python
conn = sqlitecloud.connect("sqlitecloud://cd195vehhz.g6.sqlite.cloud:8860/chinook.sqlite?apikey=cZ77GLQCoIWLbzgMwZPXTTDq8rYARAc1PnaWxn4lb8Q")
```
""")

# Add embedding stats
st.sidebar.markdown("---")
st.sidebar.subheader("Embedding Statistics")

# Count embeddings in database
embedding_count = db_adapter.execute_query("SELECT COUNT(*) FROM embeddings", fetch='one')
embedding_count = embedding_count[0] if embedding_count else 0

model_count = db_adapter.execute_query("SELECT COUNT(DISTINCT model) FROM embeddings", fetch='one')
model_count = model_count[0] if model_count else 0

chunk_count = db_adapter.execute_query("SELECT COUNT(*) FROM chunks", fetch='one')
chunk_count = chunk_count[0] if chunk_count else 0

doc_count = db_adapter.execute_query("SELECT COUNT(*) FROM documents", fetch='one')
doc_count = doc_count[0] if doc_count else 0

st.sidebar.markdown(f"""
- Total documents: {doc_count}
- Total text chunks: {chunk_count}
- Stored embeddings: {embedding_count}
- Embedding models used: {model_count}
""")