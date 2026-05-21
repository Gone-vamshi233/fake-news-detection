import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Title
st.title("Fake News Detection System")

st.write("Enter news text below to check whether it is REAL or FAKE.")

# User Input
news = st.text_area("Enter News Here")

# Prediction button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        # Transform text
        news_vec = vectorizer.transform([news])

        # Predict
        prediction = model.predict(news_vec)

        # Show result
        if prediction[0] == 1:
            st.success("REAL NEWS")
        else:
            st.error("FAKE NEWS")
