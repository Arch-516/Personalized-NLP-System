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
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8f9ff 0%,
            #eef2ff 50%,
            #f8f9ff 100%
        );
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Profile cards */
    .profile-card {
        background: white;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #eeeeee;
    }

    .profile-icon {
        font-size: 25px;
    }

    .profile-title {
        font-size: 14px;
        color: #777;
        margin-top: 5px;
    }

    .profile-value {
        font-size: 18px;
        font-weight: 700;
        margin-top: 3px;
    }

    /* Question box */
    .question-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
    }

    /* Response box */
    .response-box {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border-left: 5px solid #6c63ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.07);
        margin-top: 10px;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Intent box */
    .intent-box {
        background: #f0f1ff;
        padding: 12px 16px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Personalized NLP Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI assistant, adapted to your learning preferences ✨'
    '</div>',
    unsafe_allow_html=True
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

st.markdown("### 👤 Your Learning Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-icon">✨</div>
            <div class="profile-title">Style</div>
            <div class="profile-value">
                {profile.get("preferred_style", "Simple")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-icon">🧠</div>
            <div class="profile-title">Topic</div>
            <div class="profile-value">
                {profile.get("preferred_topic", "NLP")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-icon">📚</div>
            <div class="profile-title">Level</div>
            <div class="profile-value">
                {profile.get("preferred_level", "Beginner")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# QUESTION SECTION
# ============================================================

st.markdown(
    '<div class="question-title">💬 What would you like to learn?</div>',
    unsafe_allow_html=True
)

user_text = st.text_input(
    "",
    placeholder="Example: What is NLP?",
    label_visibility="collapsed"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🚀 Ask Assistant", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter a question first.")

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

        # ====================================================
        # RESPONSE
        # ====================================================

        st.markdown("### 🤖 Assistant Response")

        st.markdown(
            f"""
            <div class="response-box">
                {response}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # DETECTED INTENT
        # ====================================================

        st.markdown(
            f"""
            <div class="intent-box">
                🔍 <b>Detected Intent:</b> {predicted_intent}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🧠 Personalized NLP System<br>
        Powered by TF-IDF + Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)