import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    df = pd.read_csv("data/cleaned_data.csv")

    X = df['cleaned']
    y = df['sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Prediction
    y_pred = model.predict(X_test_vec)

    # Evaluation
    print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\n✅ Model & vectorizer saved!")

if __name__ == "__main__":
    train_model()