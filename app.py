import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
from PIL import Image
import io
from pikepdf import Pdf,PdfImage
import numpy as np
import pikepdf
from imgbeddings import imgbeddings

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


def get_vectorstore(text_chunks,key):
    embeddings = OpenAIEmbeddings(openai_api_key=f"{key}")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore,key):
    llm = ChatOpenAI(openai_api_key=f"{key}")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                        page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    # ibed = imgbeddings()
    key=st.text_input("Enter your Open AI  API Key")
    
    if key:

        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        st.header("Chat with multiple PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        b=st.button("Submit")
        if user_question:
            if b:
                handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    for pdf in pdf_docs:
                        pdf_read=Pdf.open(pdf)
                        for i in  range(len(pdf_read.pages)):
                            page=pdf_read.pages[i]
                            arr=list(page.images.keys())
                            if len(arr):
                                raw_image=page.images[arr[0]]
                                pdf_image=PdfImage(raw_image)
                                
                                st.image(pdf_image.as_pil_image())
                                # embedding = ibed.to_embeddings(pdf_image.as_pil_image())
                                # ar=list(embedding)
                                # st.write(ar)
                    
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks,key)
                    st.success("Done")
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore,key)
            


if __name__ == '__main__':
    main()