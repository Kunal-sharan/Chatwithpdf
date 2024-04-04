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

tut_ques=[]
template = """use this and only this {ass_chunk} chunk of text to extract the complete questions present in it
question:
"""
vector_store=None
prompt = PromptTemplate.from_template(template)
st.set_page_config(page_title="CourseMate.ai(v.0.01)",
                    page_icon=":books:")
# ibed = imgbeddings()
key=st.text_input("Enter your Open AI  API Key")

if key:
    st.header("CourseMate.ai(v.0.01) :books:")
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if  pdf_docs is not None:
        butt=st.button('Process')
        if butt:
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            if text_chunks:
                embeddings = OpenAIEmbeddings(openai_api_key=f"{key}")
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
                st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                st.success("Done") 

    tut_doc=st.file_uploader("Upload Your assignment file one at a time",accept_multiple_files=True)
    if tut_doc is not None :
        butut=st.button("Submit for Tutorial")
        if butut:
            raw_ass_text=get_pdf_text(tut_doc)
            text_ass_chunks=get_text_chunks(raw_ass_text)
            # if text_ass_chunks:
            #     ass_embeddings= OpenAIEmbeddings(openai_api_key=f"{key}")
            #     ass_store=FAISS.from_texts(text_ass_chunks,ass_embeddings)
            #     st.success("Done")
            # st.write(text_ass_chunks)
            for i in range(0,len(text_ass_chunks)):
                if text_ass_chunks[i]:
                    llm=OpenAI(openai_api_key=f"{key}")
                    chain=LLMChain(prompt=prompt,llm=llm)
                    ass_chunk=text_ass_chunks[i]
                    response=chain.run(ass_chunk)
                    # st.write(response)
                    tut_ques.append(response)
                
            st.success("Done")
                    
                    # st.write(cb)

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
                        else:
                            st.write(f"No image in the page {i+1}")
       
    

    # user_question = st.text_input("Ask a question about your documents: ")
    # b=st.button("Submit")
    newarr=[]

    with st.spinner("Processing..."):
        if 'vectorstore' in st.session_state:
            for i in range(0,len(tut_ques)):
                vectorstore=st.session_state.vectorstore
                arr=tut_ques[i].split("\n")
                st.write(arr)
                for i in range(0,len(arr)):
                    if len(arr[i])>0:
                        newarr.append(arr[i])
                        tut_topics=vectorstore.similarity_search(arr[i])
                        st.write("Question: ",arr[i])
                        st.write(tut_topics)
                        llm=OpenAI(openai_api_key=f"{key}")
                        chain=load_qa_chain(llm,chain_type="stuff")
                        with get_openai_callback() as cb:
                            response=chain.run(input_documents=tut_topics,question=arr[i])
                            st.write(response)
                            st.write(cb)
                
           
    # if user_question:
    #     if b:
    #         if  'vectorstore' in st.session_state:
    #             vec_store=st.session_state.vectorstore 
    #             docs=vectorstore.similarity_search(user_question)
    #             # st.write(docs)
    #             llm=OpenAI(openai_api_key=f"{key}")
    #             chain=load_qa_chain(llm,chain_type="stuff")
    #             with get_openai_callback() as cb:
    #                 response=chain.run(input_documents=docs,question=user_question)
    #                 st.write(response)
    #                 st.write(cb)

        