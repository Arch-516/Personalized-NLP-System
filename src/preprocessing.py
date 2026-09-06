import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Dataset path
DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "intents.csv"
)


def load_dataset():
    """Load the intent dataset."""

    data = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully!")

    print("\nDataset shape:")
    print(data.shape)

    print("\nIntent distribution:")
    print(data["intent"].value_counts())

    return data


def train_and_evaluate(data):
    """Train and evaluate the NLP model."""

    X = data["text"]
    y = data["intent"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # TF-IDF + Logistic Regression
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 45)
    print("MODEL EVALUATION")
    print("=" * 45)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return model


def main():

    print("=" * 45)
    print("       NLP MODEL TRAINING")
    print("=" * 45)

    data = load_dataset()

    train_and_evaluate(data)


if __name__ == "__main__":
    main()