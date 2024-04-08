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



def chat_with_claud(question, vectorstore, claud_key):
    if question:
        doc = vectorstore.similarity_search(question)
        chain = load_qa_chain(AnthropicLLM(model='claude-2.1', anthropic_api_key=claud_key), chain_type="stuff")
        with st.spinner(f"Searching for '{question}'"):
            response = chain.run(input_documents=doc, question=question)
            st.write("Response:", response)

st.set_page_config(page_title="CourseMate.ai(v.0.01)-->(v.0.02)", page_icon=":books:")
tut_ques=[]
template = """use this and only this {ass_chunk} chunk of text to extract the complete questions present in it
question:
"""
vector_store=None
prompt = PromptTemplate.from_template(template)
key = st.text_input("Enter your Open AI API Key")
claud_key = st.text_input("Enter your Claud AI API Key")

if key and claud_key:
    st.header("CourseMate.ai(v.0.01)-->(v.0.02) :books:")
    
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    if pdf_docs is not None:
        process_button = st.button('Process')

        if process_button:
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                if text_chunks:
                    embeddings = OpenAIEmbeddings(openai_api_key=key)
                    st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                    st.success("Text processing completed!")
            except:
                st.write("Error in reading the file.")

    option=st.selectbox(
        "What would you like to do?",
         ("Ask a Question", "Solve Assignment"),
         
         placeholder="Select which tool to use..."
    )
    if option and option == "Ask a Question" :

        if 'vectorstore' in st.session_state:
            vectorstore = st.session_state.vectorstore

            # Chatting Part
            st.subheader("Interactive Chat with with Claud AI")
            user_question = st.text_input("Ask a question:", key="user_question")

            if user_question:
                chat_with_claud(user_question, vectorstore, claud_key)
        else:
            st.write("Enter The documents First")

    if option and  option=="Solve Assignment":
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
                        st.write(tut_ques)
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

    
