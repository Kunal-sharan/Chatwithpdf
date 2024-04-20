import streamlit as  st 
import fitz  # this is pymupdf
pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'")
# Open the document
if pdf_docs :
    
    doc = fitz.open(stream=pdf_docs.read())  

    # Select the page you want to extract text from
    page = doc.load_page(4)  

    # Get the text of the page
    text = page.get_text()
    st.write(text)
    text = text.encode('ascii', 'ignore')
    # Now 'text' contains the text of the page in Unicode
    st.write(str(text))
