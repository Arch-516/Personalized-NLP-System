import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Sample text
text = "I am studying Natural Language Processing and I love learning NLP."

# 1. Convert text to lowercase
text = text.lower()

# 2. Tokenization
tokens = word_tokenize(text)

# 3. Remove stop words
stop_words = set(stopwords.words("english"))
filtered_words = [
    word for word in tokens
    if word.isalnum() and word not in stop_words
]

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_words
]

print("Original Text:")
print("I am studying Natural Language Processing and I love learning NLP.")

print("\nTokens:")
print(tokens)

print("\nAfter Stop Word Removal:")
print(filtered_words)

print("\nAfter Lemmatization:")
print(lemmatized_words)
import pandas as pd

# Load the NLP dataset
data = pd.read_csv("data/intents.csv")

print("\nDataset loaded successfully!")
print("\nFirst 5 records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nIntent distribution:")
print(data["intent"].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Prepare input and target
X = data["text"]
y = data["intent"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create NLP classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))