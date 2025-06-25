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
import pickle 
import string
# import nltk
# import os
# nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# nltk.download('punkt')
# nltk.download('stopwords')

# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
import streamlit as st
import nltk
import os

# Optional: Create a local nltk_data directory if needed
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Ensure both punkt and stopwords are downloaded properly
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)


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

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def transform_text(message):
    message = message.lower()
    message = word_tokenize(message)  # ← this is important
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
