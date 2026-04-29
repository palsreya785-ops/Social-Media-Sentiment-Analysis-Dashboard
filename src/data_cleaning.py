import pandas as pd
import re
from nltk.corpus import stopwords

# Load stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def clean_dataset():
    df = pd.read_csv("data/raw_data.csv")

    print("Before Cleaning:\n", df.head())

    df['cleaned'] = df['text'].apply(clean_text)

    print("\nAfter Cleaning:\n", df[['text', 'cleaned']].head())

    df.to_csv("data/cleaned_data.csv", index=False)
    print("\n✅ Cleaned data saved as cleaned_data.csv")

if __name__ == "__main__":
    clean_dataset()