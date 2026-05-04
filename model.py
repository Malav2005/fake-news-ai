import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Starting program...")

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Data loaded")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("Model trained")

def predict_news(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "Real News" if result == 1 else "Fake News"

print(predict_news("Breaking news: Scientists discovered new planet"))

def explain_news(text, prediction):
    if prediction == "Fake News":
        return "This news may contain misleading or exaggerated information based on patterns seen in fake news datasets."
    else:
        return "This news appears reliable and similar to verified news sources."