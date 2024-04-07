import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
# from langchain.llms import HuggingFaceHub
from PIL import Image
import io
from pikepdf import Pdf,PdfImage
import numpy as np
import pikepdf
from imgbeddings import imgbeddings
from langchain_anthropic import AnthropicLLM

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_tutorial_questions(tut_docs):
    tut_ques = []
    prompt_template = """use this and only this {ass_chunk} chunk of text to extract the complete questions present in it
    question:
    """

    for tut_doc in tut_docs:
        raw_ass_text = get_pdf_text([tut_doc])
        text_ass_chunks = get_text_chunks(raw_ass_text)

        if text_ass_chunks:
            for ass_chunk in text_ass_chunks:
                chain = LLMChain(prompt=PromptTemplate.from_template(prompt_template), llm=OpenAI(openai_api_key=key))
                response = chain.run(ass_chunk)
                tut_ques.append(response)

    return tut_ques

def chat_with_claud(question, vectorstore, claud_key):
    if question:
        doc = vectorstore.similarity_search(question)
        chain = load_qa_chain(AnthropicLLM(model='claude-2.1', anthropic_api_key=claud_key), chain_type="stuff")
        with st.spinner(f"Searching for '{question}'"):
            response = chain.run(input_documents=doc, question=question)
            st.write("Response:", response)

st.set_page_config(page_title="CourseMate.ai(v.0.01)-->(v.0.02)", page_icon=":books:")

key = st.text_input("Enter your Open AI API Key")
claud_key = st.text_input("Enter your Claud AI API Key")

if key and claud_key:
    st.header("CourseMate.ai(v.0.01)-->(v.0.02) :books:")
    
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    if pdf_docs is not None:
        process_button = st.button('Process')

        if process_button:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)

            if text_chunks:
                embeddings = OpenAIEmbeddings(openai_api_key=key)
                st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                st.success("Text processing completed!")

    
    if 'vectorstore' in st.session_state:
        vectorstore = st.session_state.vectorstore

        # Chatting Part
        st.subheader("Interactive Chat with Claud AI")
        user_question = st.text_input("Ask a question:", key="user_question")

        if user_question:
            chat_with_claud(user_question, vectorstore, claud_key)
    tut_docs = st.file_uploader("Upload Your assignment file one at a time", accept_multiple_files=True)        
    if tut_docs is not None:
        submit_button = st.button("Submit for Tutorial")

        if submit_button:
            tut_ques = generate_tutorial_questions(tut_docs)
            st.success("Tutorial questions generated successfully!")

   
