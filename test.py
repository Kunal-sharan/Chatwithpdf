# example/st_app.py

import streamlit as st
from streamlit_gsheets import GSheetsConnection
from youtubesearchpython import VideosSearch
from youtubesearchpython import Transcript
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from deep_translator import GoogleTranslator
import pyttsx3
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI

import os
# Create a TextBlob object
from youtubesearchpython import VideosSearch

# URL of the video
video_url = 'https://www.youtube.com/watch?v=z0GKGpObgPY'

# Create a VideosSearch object
videosSearch = VideosSearch(video_url)

# Fetch the video details
video_info = videosSearch.result()

# Print the video details
st.write(video_info)

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
st.header("Youtube video to translated audio converter")
arr=YouTubeTranscriptApi.get_transcript("EFg3u_E6eHU")

txt=""
for i in range(len(arr)):
    txt+=arr[i]["text"]+" "
st.write(txt)
st.write(arr)
video = Video.get('https://www.youtube.com/watch?v=z0GKGpObgPY')
st.write(video)

# Use any translator you like, in this example GoogleTranslator
g=GoogleTranslator(source='auto', target='hi')
def translate_large_text(text, chunk_size=4000):
    # Split the text into chunks of size chunk_size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    translated_text = ""
    for chunk in chunks:
        # Translate each chunk and add it to the translated text
        translated = g.translate(chunk)
        translated_text += translated
    
    return translated_text # output -> Weiter so, du bist großartig
# t=translate_large_text(txt, chunk_size=4000)
# st.write(t)
# llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyBA1JMMP5FP1gQV0UBiTIJ0Wl3QlIXOYeU')
# result = llm.invoke(f"Summarize and write this {txt} in bullets points")
# st.write(result.content)
# with st.spinner("Wait for it"):
# # Initialize the TTS engine
#     engine = pyttsx3.init()
#     tts = gTTS(t, lang='hi')
#     audio_file = "output.mp3"
#     tts.save(audio_file)

#     # Load the audio file
#     audio_bytes = open(audio_file, 'rb').read()

#     # Play the audio file with streamlit
#     st.subheader("Hindi Audio Explanation")
#     st.audio(audio_bytes, format='audio/mp3')

#     # Optionally, remove the audio file if you no longer need it
#     os.remove(audio_file)
#     # Text to be converted to speech


#     # Save the speech to an audio file
#     audio_file = "output.mp3"
#     engine.save_to_file(txt, audio_file)
#     engine.runAndWait()

#     # Load the audio file
#     audio_bytes = open(audio_file, 'rb').read()

#     # Play the audio file with streamlit
#     st.subheader("English Audio Explanation")
#     st.audio(audio_bytes, format='audio/mp3')

#     # Optionally, remove the audio file if you no longer need it
#     os.remove(audio_file)
