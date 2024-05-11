import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
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
import pytesseract
from textblob import TextBlob
from autocorrect import Speller
from spellchecker import SpellChecker
import re
from streamlit_gsheets import GSheetsConnection
from textblob import TextBlob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from transformers import *



def get_paraphrased_sentences(model, tokenizer, sentence, max_length=512, num_return_sequences=1, num_beams=5):
    # Split the sentence into chunks of max_length tokens
    sentence_chunks = [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]
    
    paraphrases = []
    for chunk in sentence_chunks:
        # Tokenize the text to be form of a list of token IDs
        inputs = tokenizer([chunk], truncation=True, padding="longest", return_tensors="pt")
        
        # Generate the paraphrased sentences
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_length  # Use max_new_tokens instead of max_length
        )
        
        # Decode the generated sentences using the tokenizer to get them back to text
        paraphrases += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return paraphrases

# sentence = "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences."
# st.write(get_paraphrased_sentences(model, tokenizer, sentence, num_beams=10, num_return_sequences=5))  


# st.dataframe(data)


spell = SpellChecker()
spell = Speller(lang='en')


# You need to replace YOUR_API_KEY with your actual API Key


# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.by import By
# from bs4 import BeautifulSoup

pytesseract.pytesseract.tesseract_cmd = r'D:\tessa\tesseract.exe'
def apply_regex_correction(extracted_text):
    try:
        # Example: Replace '0' with 'O' in the extracted text
        corrected_text = re.sub(r'0', 'O', extracted_text)
        return corrected_text
    except Exception as e:
        print("Error during regex correction:", e)
        return None

# Function to apply spell check on extracted text
def apply_spell_check(extracted_text):
    try:
        text = TextBlob(extracted_text)
        corrected_text = str(text.correct())
        return corrected_text
    except Exception as e:
        print("Error during spell check:", e)
        return None

# Function to apply language model for context-based corrections

def get_pdf_ass_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_all_images(pdf_docs,s,e):
    text=""
    for pdf in pdf_docs:
            pdf_read=Pdf.open(pdf)
            for i in  range(int(s),int(e)+1):
                page=pdf_read.pages[i]
                arr=list(page.images.keys())
                if len(arr):
                    raw_image=page.images[arr[0]]
                    pdf_image=PdfImage(raw_image)
                    img=pytesseract.image_to_string(pdf_image.as_pil_image().convert('L'))
                    text+=img
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

def create_download_link(val, filename,type_file):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file {type_file}</a>'

def chat_with_claud(question, vectorstore, claud_key,key):
    if question:
        doc = vectorstore.similarity_search(question)
        llm=OpenAI(openai_api_key=f"{key}")
        # chain = load_qa_chain(AnthropicLLM(model='claude-2.1', anthropic_api_key=claud_key), chain_type="stuff")
        chain = load_qa_chain(llm, chain_type="stuff")
        with st.spinner(f"Searching for '{question}'"):
            response = chain.run(input_documents=doc, question=question)
            st.write("Response:", response)
        return response
st.set_page_config(page_title="CourseMate.ai(v.0.01)-->(v.0.02)", page_icon=":books:",layout="wide")
conn =  st.connection("gsheets", type=GSheetsConnection)

# data = conn.read(worksheet="QnA_Data")

tut_ques=[]
imp=[]
arr_list=[]
imp_list=[]
ques_ans=[]
ans=[]
paraphrased_response=[]
p=[]
original_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
]
net_data = []
df=None
template = """use this and only this {ass_chunk} chunk of text to extract the complete questions present in it
question:
"""
vector_store=None
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
prompt = PromptTemplate.from_template(template)
api_key = st.text_input("Enter your Open AI API Key")
# claud_key = st.text_input("Enter your Claud AI API Key")
# youtube_key=st.text_input("Enter your Youtube Data V3 API key")
# api_key=st.text_input("Google ai key")

if api_key  :
    st.header("CourseMate.ai(v.0.01)-->(v.0.02) :books:")
    if 'data' not in st.session_state:
     st.session_state.data = []
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    
    if pdf_docs :
        with st.sidebar:
            opp=st.selectbox(
            "What would you like to do?",
             ( "Custom Pages","All pages(page limit)"),key="1")
            if opp and opp == "Custom Pages" :
                s=st.text_input("Enter start page ")
                e=st.text_input("Enter end page ")
                if s and e:

                    st.subheader("Your documents images")
                    for pdf in pdf_docs:
                                pdf_read=Pdf.open(pdf)
                                for i in  range(int(s),int(e)+1):
                                    page=pdf_read.pages[i]
                                    arr=list(page.images.keys())
                                    if len(arr):
                                        raw_image=page.images[arr[0]]
                                        pdf_image=PdfImage(raw_image)
                                        
                                        st.image(pdf_image.as_pil_image())
            else:
                 for pdf in pdf_docs:
                        pdf_read=Pdf.open(pdf)
                        for i in  range(len(pdf_read.pages)):
                            page=pdf_read.pages[i]
                            arr=list(page.images.keys())
                            if len(arr):
                                raw_image=page.images[arr[0]]
                                pdf_image=PdfImage(raw_image)
                                
                                st.image(pdf_image.as_pil_image())                           
        op=st.selectbox(
        "What would you like to do?",
         ( "Custom Pages","All pages","Scanned pdf with images"),key="2")
        if op and op == "Custom Pages" :
            st_page=st.text_input("Enter start page ",key="4")
            end_page=st.text_input("Enter end page ",key="3")
            if st_page and end_page:

                process_button = st.button('Process',key="b")
                if process_button:
                            try:
                                raw_text = get_pdf_text(pdf_docs,st_page,end_page)
                                text_chunks = get_text_chunks(raw_text)

                                if text_chunks:
                                    # embeddings = OpenAIEmbeddings(openai_api_key=key)
                                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=f'{api_key}')
                                    st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                                    st.success("Text processing completed!")
                            except:
                                st.write("Error in reading the file.")
        
        elif op and op == "All pages" :
                process_button = st.button('Process',key="67")
                if process_button:
                            try:
                                raw_text = get_pdf_ass_text(pdf_docs)
                                text_chunks = get_text_chunks(raw_text)

                                if text_chunks:
                                    # embeddings = OpenAIEmbeddings(openai_api_key=key)
                                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=f'{api_key}')
                                    st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                                    st.success("Text processing completed!")
                            except:
                                st.write("Error in reading the file.")
        elif op and op == "Scanned pdf with images":
            st_page=st.text_input("Enter start page ",key="5")
            end_page=st.text_input("Enter end page ",key="6")
            if st_page and end_page:
                process_button = st.button('Process',key="a")
                if process_button:
                    st.write(get_all_images(pdf_docs,st_page,end_page))
                    corrected_text_regex = apply_regex_correction(get_all_images(pdf_docs,st_page,end_page)) 

                    # Apply spell check
                    if corrected_text_regex:
                        corrected_text_spellcheck = apply_spell_check(corrected_text_regex) if corrected_text_regex else None
                        with st.container(border=True):
                         st.write(corrected_text_spellcheck)
                   
                        if corrected_text_spellcheck:
                            
                                    
                                    text_chunks = get_text_chunks(corrected_text_spellcheck)

                                    # if text_chunks:
                                    #     embeddings = OpenAIEmbeddings(openai_api_key=key)
                                    #     st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
                                    #     st.success("Text processing completed!")
                            
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

            # if user_question:
            #     text_contents=str(chat_with_claud(user_question, vectorstore, claud_key,key))
            #     st.download_button('Download some text', text_contents)
        else:
            st.write("Enter The documents First")
    "---------"
    if option and  option=="Solve Assignment":
        tut_doc=st.file_uploader("Upload Your assignment file one at a time",accept_multiple_files=True)
        if tut_doc is not None :
            butut=st.button("Submit for Tutorial")
            if butut:
                raw_ass_text=get_pdf_ass_text(tut_doc)
                # text_ass_chunks=get_text_chunks(raw_ass_text)
                # # if text_ass_chunks:
                # #     ass_embeddings= OpenAIEmbeddings(openai_api_key=f"{key}")
                # #     ass_store=FAISS.from_texts(text_ass_chunks,ass_embeddings)
                # #     st.success("Done")
                # # st.write(text_ass_chunks)
                # for i in range(0,len(text_ass_chunks)):
                #     if text_ass_chunks[i]:
                #         llm=OpenAI(openai_api_key=f"{key}")
                #         chain=LLMChain(prompt=prompt,llm=llm)
                #         ass_chunk=text_ass_chunks[i]
                #         response=chain.run(ass_chunk)
                #         # st.write(response)
                #         tut_ques.append(response)
                #         st.write(tut_ques)
                q_t=raw_ass_text.split("\n")
               
                newarr=[]
                arr=[]
                lin_yt=[]
                
                # youtube = build('youtube', 'v3', developerKey=f'{youtube_key}')              
                with st.spinner("Processing..."):
                   
                              
                        if 'vectorstore' in st.session_state:
                            # for i in range(0,len(q_t)):
                            #     vectorstore=st.session_state.vectorstore
                            #     arr=raw_ass_text().split("\n")
                            #     # st.write(arr)
                                arr=q_t
                                vectorstore=st.session_state.vectorstore
                                llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=f'{api_key}')

                                chain=load_qa_chain(llm,chain_type="stuff")
                                for i in range(0,len(arr)):
                                    if len(arr[i])>0:
                                        newarr.append(arr[i])
                                        tut_topics=vectorstore.similarity_search(arr[i])
                                        st.write("Question: ",arr[i])
                                        # st.write(tut_topics)
                                        # llm=OpenAI(openai_api_key=f"{key}")
                                        
                                        # chain = load_qa_chain(AnthropicLLM(model='claude-2.1', anthropic_api_key=claud_key), chain_type="stuff")
                                        # with get_openai_callback() as cb:
                                        try:
                                            response=chain.run(input_documents=tut_topics,question=arr[i])
                                            res_list=chain.run(input_documents=tut_topics,question="Use only and only this given chunk to provide a list of atmax 5 topics which the chunk comprises of , and make sure the 5 topics which you provide covers the entire chunk")
                                        except:
                                            response=""
                                            res_list=""    
                                            
                                            # st.write(response)
                                            ans.append(response)
                                            paraphrased_response.append(get_paraphrased_sentences(model, tokenizer,response))
                                            imp.append(res_list)
                                            
                            
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
                                        # st.write(yt_data)        
                                        lin_yt.append(yt_data)                    
                                ques_ans.append(newarr)            
                                ques_ans.append(ans)
                                ques_ans.append(imp)
                                ques_ans.append(lin_yt)
                                ques_ans.append(paraphrased_response)
                        st.session_state.data=ques_ans
                                # df=pd.DataFrame(ques_ans)
                                # st.session_state.df=df
                        
                        
                        
                        
        if "data" and "vectorstore" in st.session_state:
        # Create an "Export Report" button
            vectorstore=st.session_state.vectorstore
        # Assuming that the variables `q` and `st.session_state.data` are defined elsewhere in your code

        # Initialize an empty list to store the data

                        
                                        
            # Loop over the range of the length of `q[0]`
            if  len(st.session_state.data)>0: 
                for i in range(len(st.session_state.data[0])):
                    # For each `i`, create a dictionary with "Questions", "Answers", "Important Topics", and "Relevant Youtube Videos"
                    # Then, append this dictionary to the `net_data` list
                    for j in range(len(st.session_state.data[3][i])):
                        tut_topics=vectorstore.similarity_search(str(st.session_state.data[2][i]).split("\n")[j])
                        des=""
                        for doc in tut_topics:
                            des+=str(doc.page_content)+"\n"
                        descr=str(apply_spell_check(des))    
                        net_data.append({
                            "Questions": str(st.session_state.data[0][i]),
                            "Answers": str(st.session_state.data[1][i]),
                            "Paraphrased_answer":str(st.session_state.data[4][i]),
                            "Important Topics": str(st.session_state.data[2][i]).split("\n")[j],
                            "Topic description": str(descr),
                            "Relevant Youtube Videos Title": str(st.session_state.data[3][i][j]["title"]),
                            "Relevant Youtube Videos Link": str(st.session_state.data[3][i][j]["links"])
                        })

                # Now, `net_data` is a list of dictionaries, where each dictionary represents a data record
                df=pd.DataFrame(net_data)
                st.dataframe(df)
                # if st.button("add Report"):
                #     conn.create(worksheet="QnA_Data",data=df)
                    # st.success("Report added successfully")
                d_new=conn.read(worksheet="Paraphrased answer")
                conn.update(worksheet="CourseMate",data=df)
                st.success("Done")
                
                # if not d_new.empty:
                #     st.dataframe(d_new)
                #     d_new.dropna(subset=["Answers"], inplace=True)  # Drop rows with NaN in "Answers" column
                #     answers = df["Answers"].values  # Extract "Answers" column as array
                #     answers = answers[~pd.isnull(answers)]
                #     questions=d_new["Questions"].values
                #     questions=questions[~pd.isnull(answers)]
                #     # Remove NaN values
                #     for i in range(0,len(answers)):
                #         qu=questions[i].split("\n")
                #         q=" ".join(qu)
                #         p.append({
                #             "question":f"{q}",
                #             "orginal_answer": f"{answers[i]}",
                #             "paraphrased_answer":str(get_paraphrased_sentences(model, tokenizer, answers[i])[0])
                #         })

                #     p_frame=pd.DataFrame(p)
                #     st.dataframe(p_frame)
                #     st.write(p)
                
                #     conn.update(worksheet="Paraphrased answer",data=p_frame)
                #     d=conn.read(worksheet="Paraphrased answer")
                #     st.success("Done")
                #     st.dataframe(d)
                #     st.write(d)
                #     st.success("done")    
                if st.button("Drop Current report"):
                    conn.clear(worksheet="QnA_Data")
                    st.success("Report dropped successfully")
                if st.button("Update Current report"):
                    conn.update(worksheet="CourseMate",data=df)
                    st.success("Done")
                if st.button("Read"):
                    # conn =  st.connection("gsheets", type=GSheetsConnection)
                    d=conn.read(worksheet="CourseMate")
                    st.dataframe(d)

                            

                # If the button is clicked

                export_as_pdf = st.button("Export Report with relevant youtube videos link")
                if export_as_pdf:
                        pdf = FPDF()
                        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
                        pdf.add_font('DejaVuB', '', 'DejaVuSans-Bold.ttf', uni=True)
                        pdf.add_font('DejaVuN', '', 'DejaVuSans.ttf', uni=True)

                        for i in range(0,len(net_data)):
                            pdf.set_margins(10, 10, 5)
                            pdf.add_page()
                            report_text=""
                            links=""
                                #  links+=f"""<h1>Title:</h1><h2>{n_t}</h2>
                                #             <br>
                                #             <A HREF="{lnk}">{lnk}</A>
                                #             """ 

                            # report_text = f"Q: {str(st.session_state.data[0][i])} \n\n {str(st.session_state.data[1][i])}\n\n Important topics based on the anwsers: {str(st.session_state.data[2][i])}\n\n All Relevant Video Links: \n {links} "

                            pdf.set_font('DejaVu', '', 14)
                            # pdf.multi_cell(0,10, report_text)
                            # pdf.set_font("Arial", size = 12)
                            
                            # Section 1
                            qu=net_data[i]["Questions"].split("\n")
                            q=" ".join(qu)
                            pdf.multi_cell(0, 10, txt = "Q: " + str(q), ln = True)
                            pdf.set_font('DejaVuB', '', 12)

                            # Section 2
                            pdf.multi_cell(0, 10, txt ="A: "+ str(net_data[i]["Answers"])+"\n", ln = True)
                            pdf.set_font('DejaVuB', '', 12)
                            pdf.multi_cell(0, 10, txt ="A(1): "+ str(net_data[i]["Paraphrased_answer"])+"\n", ln = True)
                            pdf.set_font('DejaVuB', '', 12)
                            pdf.set_text_color(255, 0, 0)
                            # Section 3
                            pdf.multi_cell(0, 10, txt = "Important topics based on the answers: " , ln=True)
                            pdf.set_text_color(0, 0, 0)
                            pdf.set_font('DejaVu', '', 12)
                            important_topics = str(net_data[i]["Important Topics"])
                            pdf.multi_cell(0, 10, txt = important_topics,ln=True)
                            pdf.set_font('DejaVu', '', 12)
                            pdf.multi_cell(0,10,txt="\n"+"IMPORTANT TOPICS AND LINKS: ",ln=True)
                            
                            lnk=str(net_data[i]["Relevant Youtube Videos Link"])
                            n_t=str(net_data[i]["Relevant Youtube Videos Title"])
                            pdf.set_font('DejaVuB', '', 12)
                            pdf.multi_cell(0, 10, txt = f"{n_t}",ln=True )
                            pdf.set_font('DejaVu', '', 12)
                            pdf.set_text_color(0, 0, 255)
                            pdf.multi_cell(0, 10, txt = f"{lnk}" , ln=True)
                            pdf.set_text_color(0, 0, 0)


                                
                        html = create_download_link(pdf.output(dest="S"), "test","With youtube video links")

                        st.markdown(html, unsafe_allow_html=True)

                export_as_pdf_E = st.button("Export Report without video links")
                if export_as_pdf_E:
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
                        pdf.set_font('DejaVu', '', 14)
                        # pdf.multi_cell(0,10, report_text)
                        
                        
                        # Section 1
                        pdf.multi_cell(0, 10, txt = "Q: " + str(st.session_state.data[0][i]),ln=True)
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
                        


                            
                    html = create_download_link(pdf.output(dest="S"), "test","without video links")

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
                                        
