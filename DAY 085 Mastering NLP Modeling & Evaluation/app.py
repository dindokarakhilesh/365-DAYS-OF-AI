import streamlit as st
import pickle
import PyPDF2
import docx  # For reading .docx files
from io import BytesIO
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from contractions import fix  # To handle contractions like don't -> do not

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

nltk.download('punkt')
nltk.download('punkt_tab')  # This might be necessary if punkt_tab is specifically required
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    # Expand contractions
    text = fix(text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join tokens back to a single string
    cleaned_text = ' '.join(cleaned_tokens)

    # Remove excessive whitespaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# Function to predict sarcasm
def detect_sarcasm(new_headline):
    cleaned_headline = clean_text_advanced(new_headline)
    transformed_headline = tfidf.transform([cleaned_headline])
    prediction = model.predict(transformed_headline)

    if prediction == 1:
        return "Sarcastic"
    else:
        return "Not Sarcastic"

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to extract text from a DOCX file
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# Function to extract text from a TXT file
def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8").strip()

# Description for the app
st.sidebar.header("About the Sarcasm Detection App ")
st.sidebar.info("""
**Sarcasm Detection App** allows you to detect sarcasm in text through three methods:
1. Upload a PDF, TXT, or DOCX document, and the app will extract the text and analyze it to check for sarcasm.
2. Enter text directly into the search bar, and the app will detect whether it contains sarcasm.

**Output:** The app provides an easy-to-read prediction of whether the text is 'Sarcastic' or 'Not Sarcastic'. 
Whether you're analyzing customer reviews, social media comments, or any other text, this app simplifies sarcasm detection.
""")

# Streamlit app interface
st.title("Sarcasm Detection App")
st.markdown("This app detects **sarcasm** in text using advanced **machine learning models**. "
            "You can upload **PDF, TXT, DOCX** files or input text directly. Let's see if the text has a sarcastic tone!")
# Footer at the bottom of the sidebar
st.sidebar.markdown("---")




# File uploader for PDF, TXT, and DOCX files
uploaded_file = st.file_uploader("Upload a PDF, TXT, or DOCX file", type=["pdf", "txt", "docx"])

# Text input for the search query
search_query = st.text_input("Or, directly input the text to check for sarcasm:")

# Button to make the prediction
if st.button("Detect Sarcasm"):
    # If a PDF file is uploaded
    if uploaded_file is not None:
        # Determine file type and extract text accordingly
        if uploaded_file.type == "application/pdf":
            headline = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            headline = extract_text_from_txt(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            headline = extract_text_from_docx(uploaded_file)
        else:
            headline = ""

        if headline:
            result = detect_sarcasm(headline)
            st.success(f"Prediction (from uploaded file): **{result}**")
        else:
            st.warning("No text could be extracted from the uploaded file. Please try another file.")

    # If a search query is provided
    elif search_query:
        result = detect_sarcasm(search_query)
        st.success(f"Prediction (from input text): **{result}**")
    else:
        st.error("Please upload a file or enter some text to get a prediction.")



