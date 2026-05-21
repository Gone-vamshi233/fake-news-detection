import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Title
st.title("📰 Fake News Detection System")

# Input
news = st.text_area("Enter News Here")

# Prediction
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:

        # Convert text
        news_vec = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vec)

        # Output
        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")