import streamlit as st
# from dotenv import load_dotenv,find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_anthropic import AnthropicLLM
from PIL import Image
import io
from pikepdf import Pdf,PdfImage
import numpy as np
import pikepdf
from imgbeddings import imgbeddings
import time

ibed = imgbeddings()

st.header("CHAT WITH PDFs")
res_key=st.text_input("Put the claude api key here")
em_key=st.text_input("Put the open api key here")
if res_key and em_key:

    pdf=st.file_uploader("Upload the PDFs ", type=["pdf"])

    if pdf is not None:
            pdf_reader=PdfReader(pdf)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            # st.write(text)
            with st.sidebar:
                pdf_read=Pdf.open(pdf)
                for i in  range(len(pdf_read.pages)):
                    page=pdf_read.pages[i]
                    arr=list(page.images.keys())
                    if len(arr):
                        raw_image=page.images[arr[0]]
                        pdf_image=PdfImage(raw_image)
                        
                        st.image(pdf_image.as_pil_image())
                        embedding = ibed.to_embeddings(pdf_image.as_pil_image())
                        ar=list(embedding)
                        st.write(ar)
            text_split=CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=700,
                length_function=len
            )
            chunks=text_split.split_text(text)
            # st.write(chunks)
            
            embeddings=OpenAIEmbeddings(openai_api_key=f"{em_key}")
            # st.write(embeddings)
            knowledge_base=FAISS.from_texts(chunks,embeddings)
            
                # user_query+=" explain using the pdf and include relevant quotations taken directly from pdf in double quotes"
                
            def response_generator(user_query):
                docs=knowledge_base.similarity_search(user_query)
            # st.write(docs)
                llm=AnthropicLLM(anthropic_api_key=f"{res_key}",model='claude-2.1')
                chain=load_qa_chain(llm,chain_type="stuff")
                response = chain.run(input_documents=docs,question=user_query)
                for word in response.split():
                    yield word + " "
                    time.sleep(0.05)
            # with get_openai_callback() as cb:
            st.title("CHAT WITH YOUR PDF")

# Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("What is up?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(prompt))
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                # response=chain.run(input_documents=docs,question=user_query)
                # st.write(response)
                # st.write(cb)


