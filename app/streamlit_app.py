import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords

# Load model
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load stopwords
stop_words = set(stopwords.words('english'))

# Cleaning function (SAME as training)
def clean_text(text):
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", str(text))
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

st.title("📊 Social Media Sentiment Analysis Dashboard")
st.write("Analyze text sentiment using Machine Learning 🚀")

option = st.radio("Choose Input Type:", ["Single Text", "Upload CSV"])

# Single text
if option == "Single Text":
    text = st.text_area("Enter text")

    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            st.success(f"Predicted Sentiment: {pred}")

# CSV
else:
    file = st.file_uploader("Upload CSV")

    if file is not None:
        df = pd.read_csv(file)

        if 'cleaned' not in df.columns:
            df['cleaned'] = df.iloc[:,0].apply(clean_text)

        df['prediction'] = model.predict(vectorizer.transform(df['cleaned']))

        st.write(df.head())