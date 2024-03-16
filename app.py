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

            text_split=CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=400,
                length_function=len
            )
            chunks=text_split.split_text(text)
            # st.write(chunks)
            
            embeddings=OpenAIEmbeddings(openai_api_key=f"{em_key}")
            # st.write(embeddings)
            knowledge_base=FAISS.from_texts(chunks,embeddings)
            
            user_query=st.text_input("Ask Question about your pdf:")
            if user_query:
                user_query+=" explain using the pdf and include relevant quotations taken directly from pdf in double quotes"
                docs=knowledge_base.similarity_search(user_query)
                # st.write(docs)
                llm=AnthropicLLM(anthropic_api_key=f"{res_key}",model='claude-2.1')
                chain=load_qa_chain(llm,chain_type="stuff")
                # with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=user_query)
                st.write(response)
                # st.write(cb)


