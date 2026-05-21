import streamlit as st
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ==========================================
# LOAD MODEL & VECTORIZER
# ==========================================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==========================================
# TITLE
# ==========================================

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or headline to check whether it is REAL or FAKE."
)

# ==========================================
# USER INPUT
# ==========================================

news = st.text_area(
    "Enter News Here",
    height=200
)

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:

        # Convert text
        news_vec = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vec)

        # Show result
        if prediction[0] == 1:
            st.success("✅ REAL NEWS")

        else:
            st.error("❌ FAKE NEWS")