import joblib
import streamlit as st
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = stopwords.words('english')
ps = PorterStemmer()

def clean_emails(text: str) -> str:
  text: str = text.lower() #Lowecase
  text: str = re.sub(r'<.*?>', '', text) #remove html tags
  text: str = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text) #remove emails
  text: str = re.sub(r'https?://\S+|www\.\S+', '', text) #remove urls
  text: str = re.sub(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text) #remove phone numbers
  text: list = nltk.word_tokenize(text) #tokenize the text
  text: list = [word for word in text if word not in stop_words] #remove stopwords
  text: list = [word for word in text if word not in string.punctuation] #remove punctuation
  text: list = [ps.stem(word) for word in text] #stem words

  text: str = ' '.join(text) #join the text
  return text

model = joblib.load('model.joblib')

def spam_pred(email_text: str) -> int:
    email_text = clean_emails(email_text)
    prediction = model.predict([email_text])[0]
    return prediction

st.title('Email Spam Classifier.')
input_text = st.text_area('Enter your Email')

if st.button('Predict'):
    pred = spam_pred(input_text)

    if pred == 1:
        st.header('Email is Spam! Careful.')
    else:
        st.header('Email is not Spam.')

st.set_page_config(
    page_title="Email Spam Detection App",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Detect spam emails using a machine learning model trained with TF-IDF and Naive Bayes. Fast, accurate, and reliable."
    }
)