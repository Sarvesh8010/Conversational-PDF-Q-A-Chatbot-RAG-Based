import streamlit as st
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# Environment & Embedding Configuration
# -------------------------------------------------------------------

# Sentence-transformer based embedding model for semantic search
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------------------------------------------
# Streamlit UI Initialization
# -------------------------------------------------------------------
st.title("Q&A Chatbot With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

# -------------------------------------------------------------------
# LLM Configuration (Groq Inference Backend)
# -------------------------------------------------------------------
api_key=st.text_input("Enter your Groq API key:",type="password")

# API key is taken at runtime to avoid hardcoding secrets
if api_key:
    # Initialize Groq-hosted LLM (fast inference, production-friendly)
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")

    # Session identifier to support multiple independent chat threads
    session_id=st.text_input("Session ID",value="default_session")
    
    # Persistent in-memory store for conversational state
    if 'store' not in st.session_state:
        st.session_state.store={}

    # -------------------------------------------------------------------
    # Document Upload & Ingestion
    # -------------------------------------------------------------------
    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        
        # Convert each uploaded PDF into LangChain Document objects
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"

            # Write file to disk for loader compatibility            
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        # -------------------------------------------------------------------
        # Text Chunking & Vector Store Construction
        # -------------------------------------------------------------------

        # Spliting and creating embeddings for documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Build in-memory vector store for similarity search
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        # Retriever interface used by downstream RAG chain
        retriever = vectorstore.as_retriever()    

        # -------------------------------------------------------------------
        # History-Aware Query Reformulation
        # -------------------------------------------------------------------
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        # Enables follow-up question resolution using conversation context
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # -------------------------------------------------------------------
        # Answer Generation Prompt
        # -------------------------------------------------------------------
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use five sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

        # Full RAG pipeline = Retrieval + Generation
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        # -------------------------------------------------------------------
        # Conversational Memory Management
        # -------------------------------------------------------------------
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # -------------------------------------------------------------------
        # Chat Interface
        # -------------------------------------------------------------------
        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")