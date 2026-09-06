import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from personalization import load_profile, personalize_response


# ============================================================
# PROJECT PATHS
# ============================================================

# Find the main project folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Location of the dataset
DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "intents.csv"
)


# ============================================================
# LOAD AND TRAIN NLP MODEL
# ============================================================

def load_model():
    """Load the dataset and train the NLP model."""

    # Load dataset
    data = pd.read_csv(DATA_FILE)

    # Input text
    X = data["text"]

    # Intent labels
    y = data["intent"]

    # NLP pipeline
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

    # Train the model
    model.fit(X, y)

    return model


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    user_text,
    predicted_intent,
    profile
):
    """Generate a useful personalized response."""

    text = user_text.lower()

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if predicted_intent == "greeting":

        response = (
            "Hello! 👋 I'm your Personalized NLP Assistant. "
            "How can I help you today?"
        )

    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    elif predicted_intent in [
        "thanks",
        "thank_you"
    ]:

        response = (
            "You're welcome! 😊 I'm happy to help."
        )

    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    elif predicted_intent in [
        "goodbye",
        "bye"
    ]:

        response = (
            "Goodbye! 👋 Have a great day and keep learning!"
        )

    # --------------------------------------------------------
    # DEFINITION
    # --------------------------------------------------------

    elif predicted_intent == "definition":

        if "nlp" in text:

            response = (
                "NLP stands for Natural Language Processing. "
                "It is a branch of Artificial Intelligence "
                "that helps computers understand, process and "
                "generate human language."
            )

        elif "python" in text:

            response = (
                "Python is a high-level programming language "
                "known for its simple syntax. It is widely used "
                "in software development, data analysis, "
                "Artificial Intelligence and Machine Learning."
            )

        elif "dbms" in text or "database" in text:

            response = (
                "DBMS stands for Database Management System. "
                "It is software used to store, organize, "
                "manage and retrieve data from databases."
            )

        elif "tokenization" in text:

            response = (
                "Tokenization is the process of dividing text "
                "into smaller units called tokens. For example, "
                "a sentence can be divided into individual words."
            )

        else:

            response = (
                "A definition explains the meaning of a "
                "concept in a clear and understandable way."
            )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    elif predicted_intent == "explanation":

        if "nlp" in text:

            response = (
                "NLP allows computers to work with human "
                "language. Common NLP tasks include text "
                "classification, sentiment analysis, "
                "translation and chatbots."
            )

        else:

            response = (
                "An explanation describes how or why something "
                "works using clear and understandable information."
            )

    # --------------------------------------------------------
    # EXAMPLE
    # --------------------------------------------------------

    elif predicted_intent == "example":

        if "nlp" in text:

            response = (
                "Example of NLP: When a chatbot understands "
                "a user's question and provides a suitable "
                "answer, it is using Natural Language Processing."
            )

        elif "python" in text:

            response = (
                "Example of Python use: Python can be used "
                "to analyze a CSV dataset using the Pandas library."
            )

        else:

            response = (
                "Example: A chatbot understanding a user's "
                "question and responding to it is an example "
                "of an NLP application."
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    elif predicted_intent == "summary":

        response = (
            "Summary: NLP is a field of Artificial Intelligence "
            "that enables computers to understand and process "
            "human language. It is used in chatbots, translation, "
            "search engines and text analysis."
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    elif predicted_intent == "comparison":

        response = (
            "Comparison: Traditional programs usually follow "
            "explicit rules, while Machine Learning systems "
            "learn patterns from data. NLP applications often "
            "use Machine Learning to understand text."
        )

    # --------------------------------------------------------
    # DEFAULT RESPONSE
    # --------------------------------------------------------

    else:

        response = (
            "I understand your question, but I need a little "
            "more information to give you a better answer."
        )

    # Apply personalization
    personalized_response = personalize_response(
        response,
        profile
    )

    return personalized_response


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 50)
    print("        PERSONALIZED NLP SYSTEM")
    print("=" * 50)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading NLP model...")

    model = load_model()

    print("NLP model loaded successfully!")

    # --------------------------------------------------------
    # LOAD USER PROFILE
    # --------------------------------------------------------

    profile = load_profile()

    print("\nUser Preferences")
    print("-" * 30)

    print(
        "Response Style:",
        profile.get(
            "preferred_style",
            "simple"
        )
    )

    print(
        "Preferred Topic:",
        profile.get(
            "preferred_topic",
            "NLP"
        )
    )

    print(
        "Level:",
        profile.get(
            "preferred_level",
            "beginner"
        )
    )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    print("\nType 'exit' to stop.")

    while True:

        user_text = input("\nYou: ")

        # Exit
        if user_text.lower().strip() == "exit":

            print("\nAssistant: Goodbye! 👋")
            print("System closed.")

            break

        # Empty input
        if not user_text.strip():

            print(
                "\nAssistant: Please enter a question."
            )

            continue

        # Predict intent
        predicted_intent = model.predict(
            [user_text]
        )[0]

        # Display predicted intent
        print(
            "Detected Intent:",
            predicted_intent
        )

        # Generate response
        final_response = generate_response(
            user_text,
            predicted_intent,
            profile
        )

        # Display response
        print(
            "\nAssistant:",
            final_response
        )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()