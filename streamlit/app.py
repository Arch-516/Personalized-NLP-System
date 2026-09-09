import os
import sys
import streamlit as st

# =========================================================
# CONNECT TO SRC FOLDER
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from app import load_model, generate_response
from personalization import load_profile


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Personalized NLP Assistant",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #e9d5ff 0%,
        #ddd6fe 45%,
        #fce7f3 100%
    );
}

[data-testid="stMain"] {
    background: transparent;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #4c1d95;
    margin-top: 20px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #6b21a8;
    margin-bottom: 25px;
}

.profile-heading {
    font-size: 23px;
    font-weight: 700;
    color: #4c1d95;
    margin-top: 20px;
    margin-bottom: 15px;
}

.profile-card {
    background: rgba(255, 255, 255, 0.96);
    padding: 20px 10px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #ffffff;
    box-shadow: 0 8px 25px rgba(76, 29, 149, 0.15);
    min-height: 105px;
}

.profile-icon {
    font-size: 28px;
    margin-bottom: 5px;
}

.profile-title {
    font-size: 14px;
    color: #7e22ce;
    margin-bottom: 4px;
}

.profile-value {
    font-size: 19px;
    font-weight: 700;
    color: #4c1d95;
}

.question-title {
    font-size: 23px;
    font-weight: 700;
    color: #4c1d95;
    margin-top: 28px;
    margin-bottom: 10px;
}

div[data-testid="stTextInput"] input {
    border-radius: 14px;
    border: 2px solid #a855f7;
    padding: 14px;
    background: white;
    color: #2e1065;
    font-size: 16px;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #8b5cf6;
    opacity: 0.8;
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 12px;
    background: linear-gradient(
        90deg,
        #7c3aed,
        #c026d3
    );
    color: white;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.4);
}

.response-box {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #8b5cf6;
    box-shadow: 0 8px 25px rgba(76, 29, 149, 0.15);
    margin-top: 10px;
    font-size: 16px;
    line-height: 1.7;
    color: #2e1065;
}

.intent-box {
    background: #ede9fe;
    padding: 13px 17px;
    border-radius: 12px;
    margin-top: 15px;
    color: #5b21b6;
    border: 1px solid #ddd6fe;
}

.footer {
    text-align: center;
    color: #6b21a8;
    font-size: 13px;
    margin-top: 35px;
    padding-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Personalized NLP Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI assistant, adapted to your learning preferences ✨</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# LOAD MODEL AND PROFILE
# =========================================================

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_profile():
    return load_profile()


model = get_model()
profile = get_profile()


# =========================================================
# USER PROFILE
# =========================================================

st.markdown(
    '<div class="profile-heading">👤 Your Learning Profile</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:
    st.markdown(
        '<div class="profile-card">'
        '<div class="profile-icon">✨</div>'
        '<div class="profile-title">Style</div>'
        '<div class="profile-value">'
        + str(profile.get("preferred_style", "Simple"))
        + '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col2:
    st.markdown(
        '<div class="profile-card">'
        '<div class="profile-icon">🧠</div>'
        '<div class="profile-title">Topic</div>'
        '<div class="profile-value">'
        + str(profile.get("preferred_topic", "NLP"))
        + '</div>'
        '</div>',
        unsafe_allow_html=True
    )


with col3:
    st.markdown(
        '<div class="profile-card">'
        '<div class="profile-icon">📚</div>'
        '<div class="profile-title">Level</div>'
        '<div class="profile-value">'
        + str(profile.get("preferred_level", "Beginner"))
        + '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# QUESTION
# =========================================================

st.markdown(
    '<div class="question-title">💬 What would you like to learn?</div>',
    unsafe_allow_html=True
)

user_text = st.text_input(
    "",
    placeholder="Example: What is NLP?",
    label_visibility="collapsed"
)


# =========================================================
# ASK ASSISTANT
# =========================================================

if st.button("🚀 Ask Assistant", use_container_width=True):

    if user_text.strip() == "":
        st.warning("⚠️ Please enter a question first.")

    else:

        predicted_intent = model.predict([user_text])[0]

        response = generate_response(
            user_text,
            predicted_intent,
            profile
        )

        st.markdown("### 🤖 Assistant Response")

        st.markdown(
            '<div class="response-box">'
            + str(response).replace("\n", "<br>")
            + '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="intent-box">'
            '🔍 <b>Detected Intent:</b> '
            + str(predicted_intent)
            + '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🧠 Personalized NLP System'
    '<br><br>'
    'Powered by <b>TF-IDF + Logistic Regression</b> ✨'
    '</div>',
    unsafe_allow_html=True
)