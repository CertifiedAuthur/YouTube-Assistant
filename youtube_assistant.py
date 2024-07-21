import streamlit as st
import textwrap
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS

# Initialize OpenAI embeddings
def initialize_openai(api_key: str):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    llm = OpenAI(api_key=api_key)
    return embeddings, llm

def create_vector_db(video_url: str, api_key: str) -> FAISS:
    """Create a vector database from YouTube video transcripts."""
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
    except ImportError as e:
        st.error(f"Error loading YouTube transcript: {e}")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(transcript)
    
    embeddings, _ = initialize_openai(api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db: FAISS, query: str, api_key: str, k: int = 4) -> str:
    """Generate a response to the query based on the vector database."""
    try:
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
        
        _, llm = initialize_openai(api_key)
        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.
            
            Answer the following question: {question}
            By searching the following video transcript: {docs}
            
            Only use the factual information from the transcript to answer the question.
            
            If you feel like you don't have enough information to answer the question, say "I don't know".
            
            Your answers should be detailed.
            """
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."

# --- Streamlit App ---
st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.text_area(
            label="What is the YouTube video URL?",
            max_chars=50
        )
        query = st.text_area(
            label="Ask me about the video?",
            max_chars=50,
            key="query"
        )
        api_key = st.text_area(
            label="Input your OpenAI API Key",
            max_chars=56
        )
        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url and api_key:
    db = create_vector_db(youtube_url, api_key)
    if db:
        response = get_response_from_query(db, query, api_key)
        st.subheader("Answer")
        st.text(textwrap.fill(response, width=80))
