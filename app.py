import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
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


st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")
st.write(css, unsafe_allow_html=True)
# ibed = imgbeddings()
key=st.text_input("Enter your Open AI  API Key")

if key:
    st.header("Chat with multiple PDFs :books:")
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if  pdf_docs is not None:
        raw_text = get_pdf_text(pdf_docs)

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        if text_chunks:
            embeddings = OpenAIEmbeddings(openai_api_key=f"{key}")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) 
        st.success("Done")   
    with st.spinner("Processing"):
            # get pdf text
            with st.sidebar:
                st.header("Image Extracted from Docs: ")

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
            
            
            
    user_question = st.text_input("Ask a question about your documents:")
    b=st.button("Submit")
    if user_question:
        if b:
            user_ques=f"{user_question} explain and include relevant quotations taken directly from pdfs in double quotes"
            docs=vectorstore.similarity_search(user_ques)
            # st.write(docs)
            llm=OpenAI(openai_api_key=f"{key}")
            chain=load_qa_chain(llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=user_ques)
                st.write(response)
                st.write(cb)

        