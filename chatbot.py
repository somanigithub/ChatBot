import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#upload pdf files
OPENAI_API_KEY="sk-proj-1i1gNxLa2Ji3LwmSF10MT3BlbkFJMC0rZN0nztPtnvoEUaD1"
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file= st.file_uploader("Upload your pdf file and start asking questions", type="pdf")

#extract the text

if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators='\n',
        chunk_size=1000,
        chunk_overlap=150,
        length_function = len
    )

    chunks= text_splitter.split_text(text)
    #st.write(chunks)

    #generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store- FAISS
    vector_store= FAISS.from_texts(chunks, embeddings)

    #get user's question
    user_question=st.text_input("Type your question here")

    #do a similarity search in vector store
    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)
