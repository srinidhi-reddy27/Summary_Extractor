import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF for PDF processing
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from summa.summarizer import summarize
from io import BytesIO
from PIL import Image
import docx
import requests
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset for training
dataset_path = "./Dataset_MAIN.xlsx"
df = pd.read_excel(dataset_path, sheet_name="Sheet1")

df.drop_duplicates(inplace=True)
df.columns = df.columns.str.strip().str.lower()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    else:
        text = ""
    return text

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df["abstract"].astype(str).apply(preprocess_text))
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    
    top_keywords = feature_array[tfidf_sorting][:top_n]
    
    # Ensure multi-word phrases are correctly formatted
    formatted_keywords = []
    for keyword in top_keywords:
        formatted_keyword = ' '.join([word.capitalize() for word in keyword.split()])
        formatted_keywords.append(formatted_keyword)
    
    return formatted_keywords


def get_topic_from_lda(text):
    processed_text = preprocess_text(text)
    X_new = vectorizer.transform([processed_text])
    topic_distribution = lda.transform(X_new)[0]
    keywords = extract_keywords(processed_text)
    return f"{', '.join(keywords)}"

def is_logo(image):
    width, height = image.size
    aspect_ratio = width / height
    
    if width < 300 and height < 300:
        return True  # Small image
    
    if 0.8 < aspect_ratio < 1.2:
        return True  # Square aspect ratio (common for logos)
    
    return False

def extract_images_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for page in doc:
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_pil = Image.open(BytesIO(img_bytes))
            if not is_logo(img_pil):
                images.append(img_pil)
    return images

def extract_images_from_docx(file):
    doc = docx.Document(file)
    images = []
    for rel in doc.part.rels:
        if "image" in doc.part.rels[rel].target_ref:
            img_bytes = doc.part.rels[rel].target_part.blob
            img_pil = Image.open(BytesIO(img_bytes))
            if not is_logo(img_pil):
                images.append(img_pil)
    return images



# Streamlit GUI
st.title("Unsupervised Topic Modeling & Summarization of Scientific Research Documents")
st.write("Upload a document (PDF, DOCX, TXT) or enter a URL below.")

input_type = st.selectbox('Select the document type', ['PDF', 'DOCX', 'Text File', 'Direct Text', 'URL'])

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join([para.get_text() for para in paragraphs if para.get_text()])

text = ""
images = []
if input_type == 'PDF':
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        text = extract_text_from_pdf(file_bytes)
        images = extract_images_from_pdf(file_bytes)
elif input_type == 'DOCX':
    uploaded_file = st.file_uploader("Upload your DOCX file", type=["docx"])
    if uploaded_file:
        text = extract_text_from_docx(uploaded_file)
        images = extract_images_from_docx(uploaded_file)
elif input_type == 'Text File':
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode('utf-8')
elif input_type == 'Direct Text':
    text = st.text_area("Enter text directly", height=300)
elif input_type == 'URL':
    url = st.text_input("Enter the link to the document (PDF, DOCX, or Webpage)")
    if url:
        text = fetch_text_from_url(url)

if text:
    topic = get_topic_from_lda(text)
    summary = summarize(text, ratio=0.2)
    
    st.subheader(f"Predicted Topic: {topic}")
    st.subheader("Extracted Text")
    st.text_area("Extracted Text", text, height=300)
    
    st.subheader("Summary")
    st.markdown(f"<div style='text-align: justify; font-size: 16px;'>{summary}</div>", unsafe_allow_html=True)
    
    if images:
        st.subheader("Extracted Images")
        for img in images:
            st.image(img, use_container_width=True)
