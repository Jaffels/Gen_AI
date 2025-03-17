import streamlit as st
import os
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

# Initialize session state for storing conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

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

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    process_button = st.button("Process Documents")


# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        # Create a temporary file to handle the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name

        try:
            pdf_reader = PdfReader(temp_file_path)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    return text


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
        llm = ChatOpenAI(model=model_option, temperature=temperature)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
        extracted_text = extract_text_from_pdfs(uploaded_files)

        if extracted_text:
            # Create vector store
            st.session_state.vector_store = create_vector_store(extracted_text)

            if st.session_state.vector_store:
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                st.success(f"Successfully processed {len(uploaded_files)} document(s)!")

                # Clear previous messages when new documents are processed
                st.session_state.messages = []
        else:
            st.warning("No text could be extracted from the uploaded documents.")

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
    """)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) to:
    - Extract text from uploaded PDFs
    - Create embeddings of document chunks
    - Retrieve relevant context from your documents
    - Generate responses based on the document content
    """)