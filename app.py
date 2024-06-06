import os
import io
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from PyPDF2 import PdfReader
import fitz  # this is pymupdf

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_resource
def get_gemini_response(prompt):
    #Modelin Ayar Kısmı
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }


    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    prompt_token_count = model.count_tokens(prompt)

    response = model.generate_content(prompt).text

    yanıt = st.markdown(response)

    response_token_count = model.count_tokens(response)

    return yanıt, prompt_token_count, response_token_count



@st.cache_resource
def read_pdf(file):
    pdfReader = PdfReader(file)
    count = len(pdfReader.pages)

    all_page_text = ""

    for i in range(count):  # for i in range (0, count-1) ///  for i in range (len(pdfReader.pages))
        page = pdfReader.pages[i]
        all_page_text += page.extract_text()

    return all_page_text



@st.cache_resource
def read_pdf_2(file_path):
    doc = fitz.open(file_path)
    images = []
    # count = len(doc)
    for i in range(len(doc)):  # for i in range(count)
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

st.set_page_config(page_title="ATS Sistemi",
                   page_icon="🧠",
                   initial_sidebar_state="expanded"
                   )

st.sidebar.header("ATS Sistemimize Hoşgeldiniz")



