import streamlit as  st 
import time

@st.cache_data
def long_running_function():
    time.sleep(5)  # This could be a long running computation
    return "Expensive Result"

result = long_running_function()
st.write(result)
