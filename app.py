# import streamlit as st
# import pickle 
# import string
# # necessary libraries for text transforming func
# import nltk 
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer


# ps = PorterStemmer()

# def transform_text(message):
#     message = message.lower()  # Lowercase conversion
#     message = nltk.word_tokenize(message)  # Tokenization
#     y = []
#     for i in message:
#         if i.isalnum():
#             y.append(i)
#     message = y[:]
#     y.clear() # Clear the list for stopwords and punctuation removal
#     for i in message:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     message = y[:]
#     y.clear()  # Clear the list for stemming
#     for i in message:
#         y.append(ps.stem(i))
#     return " ".join(y)

# # Load the pre-trained model
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message...")

# if st.button("Predict"):
#     # 1. Preprocess the input
#     transform_sms = transform_text(input_sms)
#     # 2. Transform the input using the TF-IDF vectorizer
#     input_data_features = tfidf.transform([transform_sms])
#     # 3. Predict using the loaded model
#     prediction = model.predict(input_data_features)[0]
#     # 4. Display the prediction result
#     if prediction == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

import streamlit as st
import os
import nltk
import pickle
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up nltk data path and download resources if needed
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

for res in ['punkt', 'stopwords']:
    try:
        if res == 'punkt':
            nltk.data.find('tokenizers/punkt')
        else:
            nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download(res, download_dir=nltk_data_path)

# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# Minimal styling using markdown
st.markdown("""
    <style>
        .main {
            padding: 20px;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #2c3e50;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: #e74c3c;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

ps = PorterStemmer()

def transform_text(message):
    message = message.lower()
    message = word_tokenize(message)  # <- uses punkt safely
    y = []
    for i in message:
        if i.isalnum():
            y.append(i)
    message = y[:]
    y.clear()
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    message = y[:]
    y.clear()
    for i in message:
        y.append(ps.stem(i))
    return " ".join(y)


# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message...")

if st.button("Predict"):
    transform_sms = transform_text(input_sms)
    input_data_features = tfidf.transform([transform_sms])
    prediction = model.predict(input_data_features)[0]
    
    if prediction == 1:
        st.markdown('<div class="result">🚫 Spam</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result">✅ Not Spam</div>', unsafe_allow_html=True)
