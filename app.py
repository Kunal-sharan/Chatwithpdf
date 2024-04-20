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
from fpdf import FPDF
import base64
from io import BytesIO
from youtubesearchpython import VideosSearch
# from langchain_google_genai import GoogleGenerativeAI

# You need to replace YOUR_API_KEY with your actual API Key


# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.by import By
# from bs4 import BeautifulSoup


def get_pdf_ass_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_pdf_text(pdf_docs,st,end):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in range(int(st),(int(end)+1)):
            text += pdf_reader.pages[page].extract_text()
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

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

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
ques_ans=[]
ans=[]
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
    
    if pdf_docs :
        op=st.selectbox(
        "What would you like to do?",
         ( "Custom Pages","All pages(page limit)"))
        if op and op == "Custom Pages" :
            st_page=st.text_input("Enter start page ")
            end_page=st.text_input("Enter end page ")
            if st_page and end_page:

                process_button = st.button('Process')
                if process_button:
                            try:
                                raw_text = get_pdf_text(pdf_docs,st_page,end_page)
                                text_chunks = get_text_chunks(raw_text)

                                if text_chunks:
                                    embeddings = OpenAIEmbeddings(openai_api_key=key)
                                    st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                                    st.success("Text processing completed!")
                            except:
                                st.write("Error in reading the file.")
        
        if op and op == "All pages(page limit 20 pages)" :
                process_button = st.button('Process')
                if process_button:
                            try:
                                raw_text = get_pdf_ass_text(pdf_docs)
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
                raw_ass_text=get_pdf_ass_text(tut_doc)
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
                
                lin_yt=[]
                # youtube = build('youtube', 'v3', developerKey=f'{youtube_key}')              
                with st.spinner("Processing..."):
                    if 'vectorstore' in st.session_state:
                        for i in range(0,len(tut_ques)):
                            vectorstore=st.session_state.vectorstore
                            arr=tut_ques[i].split("\n")
                            # st.write(arr)
    
                            for i in range(0,len(arr)):
                                if len(arr[i])>0:
                                    newarr.append(arr[i])
                                    tut_topics=vectorstore.similarity_search(arr[i])
                                    st.write("Question: ",arr[i])
                                    # st.write(tut_topics)
                                    llm=OpenAI(openai_api_key=f"{key}")
                                    # llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=f'{api_key}')

                                    chain=load_qa_chain(llm,chain_type="stuff")
                                    # chain = load_qa_chain(AnthropicLLM(model='claude-2.1', anthropic_api_key=claud_key), chain_type="stuff")
                                    with get_openai_callback() as cb:
                                        try:
                                            response=chain.run(input_documents=tut_topics,question=arr[i])
                                            res_list=chain.run(input_documents=tut_topics,question="Use only and only this given chunk to provide a list of atmax 5 topics which the chunk comprises of , and make sure the 5 topics which you provide covers the entire chunk")
                                        except:
                                            response="No data"
                                            res_list=""
                                        st.write(response)
                                        ans.append(response)
                                        imp.append(res_list)
                                        st.write(cb)
                        
                                    arr_list=res_list.split("\n")
                                    # for j in range(0,len(arr_list)):
                                    #     imp_list.append(arr_list[j])   
                                # st.write(imp_list)
                                
                                    # st.write(arr_list)
                                    yt_data=[]
                                    for i in range(0,len(arr_list)):
                                        if len(arr_list[i])>0:
                                            
                                            
                                            # search_response = youtube.search().list(
                                            #             q=f'{arr_list[i]}',
                                            #             part='id,snippet',
                                            #             maxResults=1
                                            #         ).execute()
                                            # for search_result in search_response.get('items', []):
                                            #     if search_result['id']['kind'] == 'youtube#video':
                                            #         video_data = {
                                            #                 "topic":f"{arr_list[i]}",
                                            #                 "title": f"{search_result['snippet']['title']}",
                                            #                 "links": f"https://www.youtube.com/watch?v={search_result['id']['videoId']}",
                                            #                 "desc":f"Description: {search_result['snippet']['description']}"
                                            #             }
                                            videoSearch=VideosSearch(f"{arr_list[i]}",limit=1)
                                            if videoSearch.result()['result']:
                                                video_data={
                                                    "title":f"{str(videoSearch.result()['result'][0]['title'])}",
                                                    "links":f"{videoSearch.result()['result'][0]['link']}",
                                                    "thumbnail":f"{videoSearch.result()['result'][0]['thumbnails'][0]['url']}"
                                                }
                                                yt_data.append(video_data)
                                    st.write(yt_data)        
                                    lin_yt.append(yt_data)                    
                            ques_ans.append(newarr)            
                            ques_ans.append(ans)
                            ques_ans.append(imp)
                            ques_ans.append(lin_yt)
                            st.session_state.data=ques_ans
                            
        if "data" in st.session_state:
            # Create an "Export Report" button
            export_as_pdf = st.button("Export Report")
            q=st.session_state.data
            
# If the button is clicked
            if export_as_pdf:
                # st.write(q)
                # st.write(len(ques_ans))
                pdf = FPDF()
                pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                pdf.add_font('DejaVuB', '', 'DejaVuSans-Bold.ttf', uni=True)
                pdf.add_font('DejaVuN', '', 'DejaVuSans.ttf', uni=True)

                for i in range(0,len(st.session_state.data[0])):
                    pdf.set_margins(10, 10, 5)
                    pdf.add_page()
                    report_text=""
                    links=""
                        #  links+=f"""<h1>Title:</h1><h2>{n_t}</h2>
                        #             <br>
                        #             <A HREF="{lnk}">{lnk}</A>
                        #             """ 

                    report_text = f"Q: {str(st.session_state.data[0][i])} \n\n {str(st.session_state.data[1][i])}\n\n Important topics based on the anwsers: {str(st.session_state.data[2][i])}\n\n All Relevant Video Links: \n {links} "
#                     report_html="""<h1>Question</h1>
# <p>{str(st.session_state.data[0][i])}</p>

# <h1>Answer</h1>
# <p>{str(st.session_state.data[1][i])}</p>

# <h1>Important Topics</h1>
# <ul>
#     <li> {str(st.session_state.data[2][i])}</li>
    
# </ul>

# <h1>Video Links</h1>
# <ul>
#     <li><a href="[1](https://www.youtube.com/watch?v=dQw4w9WgXcQ)">Video 1</a></li>
#     <li><a href="[2](https://www.youtube.com/watch?v=3tmd-ClpJxA)">Video 2</a></li>
#     <li><a href="[3](https://www.youtube.com/watch?v=ktvTqknDobU)">Video 3</a></li>
# </ul>
# """
                    pdf.set_font('DejaVu', '', 14)
                    # pdf.multi_cell(0,10, report_text)
                    # pdf.set_font("Arial", size = 12)
                    
                    # Section 1
                    pdf.multi_cell(0, 10, txt = "Q: " + str(st.session_state.data[0][i]), ln = True)
                    pdf.set_font('DejaVuB', '', 12)

                    # Section 2
                    pdf.multi_cell(0, 10, txt ="A: "+ str(st.session_state.data[1][i])+"\n", ln = True)
                    pdf.set_font('DejaVuB', '', 14)
                    pdf.set_text_color(255, 0, 0)
                    # Section 3
                    pdf.multi_cell(0, 10, txt = "Important topics based on the answers: " , ln=True)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font('DejaVu', '', 12)
                    important_topics = str(st.session_state.data[2][i])
                    pdf.multi_cell(0, 10, txt = important_topics,ln=True)
                    pdf.set_font('DejaVu', '', 14)
                    pdf.multi_cell(0,10,txt="\n"+"IMPORTANT TOPICS AND LINKS: ",ln=True)
                    for j in range(0,len(st.session_state.data[3][i])):
                        title=st.session_state.data[3][i][j]["title"]
                        t_arr=title.split(" ")
                        n_t=' '.join(map(str, t_arr))
                        lnk=st.session_state.data[3][i][j]["links"]
                        pdf.set_font('DejaVuB', '', 12)
                        pdf.multi_cell(0, 10, txt = f"{n_t}",ln=True )
                        pdf.set_font('DejaVu', '', 12)
                        pdf.set_text_color(0, 0, 255)
                        pdf.multi_cell(0, 10, txt = f"{lnk}" , ln=True)
                        pdf.set_text_color(0, 0, 0)


                         
                html = create_download_link(pdf.output(dest="S"), "test")

                st.markdown(html, unsafe_allow_html=True)

                        
                        # num_rows = len(yt_data) // 1
                        # if len(yt_data) % 1 != 0:
                        #     num_rows += 1

                        # for i in range(num_rows):
                        #     cols = st.columns(1)
                        #     for j in range(1):
                        #         index = i * 1 + j
                        #         if index < len(yt_data):
                        #             tile = cols[j].container(height=700)
                        #             tile.write(yt_data[index]["topic"])
                        #             tile.write(yt_data[index]["title"])
                        #             tile.video(yt_data[index]["links"])
                        #             tile.write(yt_data[index]["desc"])
                                

                        
