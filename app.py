
import streamlit as st
import joblib
import re
import html

# Load final model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("final_model.pkl")
    vectorizer = joblib.load("final_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Page configuration
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fake Job Posting Detection System")
st.write(
    "An AI-powered application that analyzes job postings "
    "and predicts whether they are genuine or fraudulent."
)

st.divider()

st.subheader("Enter Job Posting Details")

title = st.text_input("Job Title")
company = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")
benefits = st.text_area("Benefits")

if st.button("Analyze Job Posting", type="primary"):

    if not any([title, company, description, requirements, benefits]):
        st.warning("Please enter at least some job details.")

    else:
        raw_text = " ".join([
            title, company, description,
            requirements, benefits
        ])

        # Same basic text normalization used during training
        text = html.unescape(raw_text)
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Potentially Fraudulent Job Posting")
            st.write(
                "The model identifies this posting as potentially fraudulent. "
                "Verify the employer before sharing personal information or paying fees."
            )
        else:
            st.success("✅ Likely Genuine Job Posting")
            st.write(
                "The model identifies this posting as likely genuine. "
                "Still verify the employer and job details independently."
            )

        st.caption(
            "Disclaimer: This is an ML prediction, not a guarantee. "
            "Always verify job postings independently."
        )

st.divider()
st.caption("Fake Job Posting Detection | Machine Learning Project")
