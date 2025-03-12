import streamlit as st
import openai
import faiss
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Set OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# Load or create FAISS index
def load_vectorstore():
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except FileNotFoundError:
        return None


def create_vectorstore():
    loader = TextLoader("data.txt")  # Load documents
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    vectorstore.save_local("faiss_index")

    return vectorstore


# Load or initialize vector store
vectorstore = load_vectorstore() or create_vectorstore()


# RAG-based response function
def get_response(query):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-4o-mini"), retriever=retriever)
    response = qa_chain.run(query)
    return response


# Streamlit UI
st.title("RAG-Powered Chatbot")
st.write("Ask me anything!")

user_query = st.text_input("Your Question:")

if user_query:
    response = get_response(user_query)
    st.write("### Response:")
    st.write(response)