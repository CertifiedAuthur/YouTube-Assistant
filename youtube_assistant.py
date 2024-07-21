import streamlit as st
import textwrap
from langchain.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, api_key, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = OpenAI(openai_api_key = api_key)
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful Youtube assistant that can answer questions about videos based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be detailed.
        """
    )
    
    sequence = prompt | llm
    
    response = sequence.invoke({"question": query, "docs": docs_page_content})
    response = response.replace("\n", "")
    return response

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
            label ="Input your OpenAI API Key",
            max_chars=56
        )
        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url and api_key:
    db = create_vector_db(youtube_url)
    response = get_response_from_query(db, query, api_key)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width=80))
