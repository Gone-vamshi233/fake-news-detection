import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article or headline to check whether it is REAL or FAKE."
)

# Input
news = st.text_area(
    "Enter News Here",
    height=200
)

# Predict
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:

        # Transform
        news_vec = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vec)

        # Output
        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")