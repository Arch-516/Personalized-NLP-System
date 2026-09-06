import os
import sys
import streamlit as st

# ============================================================
# CONNECT TO SRC FOLDER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, SRC_DIR)

from app import load_model, generate_response
from personalization import load_profile


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Personalized NLP Assistant",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Personalized NLP Assistant")

st.write(
    "Ask a question and get a personalized NLP response."
)

st.divider()


# ============================================================
# LOAD MODEL AND PROFILE
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_profile():
    return load_profile()


model = get_model()
profile = get_profile()


# ============================================================
# USER PROFILE
# ============================================================

st.subheader("👤 User Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Style",
        profile.get("preferred_style", "simple")
    )

with col2:
    st.metric(
        "Topic",
        profile.get("preferred_topic", "NLP")
    )

with col3:
    st.metric(
        "Level",
        profile.get("preferred_level", "beginner")
    )


st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

st.subheader("💬 Ask your question")

user_text = st.text_input(
    "Enter your question:",
    placeholder="Example: What is NLP?"
)


# ============================================================
# GENERATE RESPONSE
# ============================================================

if st.button("🚀 Ask Assistant"):

    if user_text.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Predict intent
        predicted_intent = model.predict(
            [user_text]
        )[0]

        # Generate personalized response
        response = generate_response(
            user_text,
            predicted_intent,
            profile
        )

        st.success("Response generated!")

        st.subheader("🤖 Assistant")

        st.write(response)

        st.divider()

        st.write(
            "🔍 Detected Intent:",
            predicted_intent
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Personalized NLP System | "
    "TF-IDF + Logistic Regression"
)