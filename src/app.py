import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from personalization import load_profile, personalize_response


# Load dataset
data = pd.read_csv("data/intents.csv")

X = data["text"]
y = data["intent"]


# Create and train NLP model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)


# Load user's preferences
profile = load_profile()


print("===================================")
print("      PERSONALIZED NLP SYSTEM")
print("===================================")

print("Response Style:", profile["preferred_style"])
print("Preferred Topic:", profile["preferred_topic"])
print("Level:", profile["preferred_level"])

print("\nType 'exit' to stop.")


def generate_response(user_text, predicted_intent, profile):

    text = user_text.lower()

    # Greeting
    if predicted_intent == "greeting":
        response = "Hello! How can I help you today?"

    # Thanks
    elif predicted_intent in ["thanks", "thank_you"]:
        response = "You're welcome! I'm happy to help."

    # Goodbye
    elif predicted_intent in ["goodbye", "bye"]:
        response = "Goodbye! Have a great day."

    # Definition
    elif predicted_intent == "definition":

        if "nlp" in text:
            response = (
                "NLP stands for Natural Language Processing. "
                "It allows computers to understand and process human language."
            )

        elif "python" in text:
            response = (
                "Python is a high-level programming language "
                "used for software development, data analysis, "
                "machine learning and many other applications."
            )

        elif "dbms" in text or "database" in text:
            response = (
                "DBMS stands for Database Management System. "
                "It is software used to store, manage and retrieve data efficiently."
            )

        elif "tokenization" in text:
            response = (
                "Tokenization is the process of breaking text "
                "into smaller units called tokens."
            )

        else:
            response = (
                f"You are asking for a definition related to "
                f"{profile['preferred_topic']}."
            )

    # Default response
    else:
        response = (
            f"I understand that you are asking about "
            f"{predicted_intent}."
        )

    return personalize_response(response, profile)


# Main conversation loop
while True:

    user_text = input("\nYou: ")

    if user_text.lower() == "exit":
        print("System closed.")
        break

    # Predict intent
    predicted_intent = model.predict([user_text])[0]

    print("Detected Intent:", predicted_intent)

    # Generate personalized response
    final_response = generate_response(
        user_text,
        predicted_intent,
        profile
    )

    print("\nAssistant:", final_response)