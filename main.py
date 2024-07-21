import streamlit as st
import youtubeloader as yt
import textwrap

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
            label ="Input your OpenAI_API_KEY",
            max_chars=56
        )
        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url and api_key:
    db = yt.create_vector_db(youtube_url)
    response = yt.get_response_from_query(db, query, api_key)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width=80))