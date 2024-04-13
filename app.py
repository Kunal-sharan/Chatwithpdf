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
from googleapiclient.discovery import build
# from langchain_google_genai import GoogleGenerativeAI

# You need to replace YOUR_API_KEY with your actual API Key


# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.by import By
# from bs4 import BeautifulSoup


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
imp=[]
arr_list=[]
imp_list=[]
template = """use this and only this {ass_chunk} chunk of text to extract the complete questions present in it
question:
"""
vector_store=None
prompt = PromptTemplate.from_template(template)
key = st.text_input("Enter your Open AI API Key")
claud_key = st.text_input("Enter your Claud AI API Key")
youtube_key=st.text_input("Enter your Youtube Data V3 API key")
# api_key=st.text_input("Google ai key")

if key and claud_key and youtube_key :
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
         ( "Solve Assignment","Ask a Question"),
         
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
                                    # st.write(tut_topics)
                                    llm=OpenAI(openai_api_key=f"{key}")
                                    # llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=f'{api_key}')

                                    chain=load_qa_chain(llm,chain_type="stuff")
                                    with get_openai_callback() as cb:
                                        try:
                                            response=chain.run(input_documents=tut_topics,question=arr[i])
                                            res_list=chain.run(input_documents=tut_topics,question="Use only and only this given chunk to provide a list of atmax 5 topics which the chunk comprises of , and make sure the 5 topics which you provide covers the entire chunk")
                                        except:
                                            response="No data"
                                            res_list=""
                                        st.write(response)
                                        imp.append(res_list)
                                        st.write(cb)
                        # st.write(imp)
                        for i in range(0,len(imp)):
                            arr_list=imp[i].split("\n")
                            for j in range(0,len(arr_list)):
                                imp_list.append(arr_list[j])   
                        # st.write(imp_list)
                        youtube = build('youtube', 'v3', developerKey=f'{youtube_key}')
                        yt_data=[]
                        for i in range(0,len(imp_list)):
                            if len(imp_list[i])>0:
                                
                                
                                search_response = youtube.search().list(
                                            q=f'{imp_list[i]}',
                                            part='id,snippet',
                                            maxResults=1
                                        ).execute()
                                for search_result in search_response.get('items', []):
                                    if search_result['id']['kind'] == 'youtube#video':
                                        video_data = {
                                                "topic":f"{imp_list[i]}",
                                                "title": f"{search_result['snippet']['title']}",
                                                "links": f"https://www.youtube.com/watch?v={search_result['id']['videoId']}",
                                                "desc":f"Description: {search_result['snippet']['description']}"
                                            }
                                        yt_data.append(video_data)
                        num_rows = len(yt_data) // 1
                        if len(yt_data) % 1 != 0:
                            num_rows += 1

                        for i in range(num_rows):
                            cols = st.columns(1)
                            for j in range(1):
                                index = i * 1 + j
                                if index < len(yt_data):
                                    tile = cols[j].container(height=700)
                                    tile.write(yt_data[index]["topic"])
                                    tile.write(yt_data[index]["title"])
                                    tile.video(yt_data[index]["links"])
                                    tile.write(yt_data[index]["desc"])
                                

                        
