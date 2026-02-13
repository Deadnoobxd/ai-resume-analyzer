import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🧠 AI Resume Analyzer")
st.write("Upload your resume and compare it with a job description.")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze Resume"):
    if uploaded_file is not None and job_description != "":
        
        resume_text = extract_text_from_pdf(uploaded_file)

        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        # Vectorization
        cv = CountVectorizer().fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(cv)[0][1]

        match_percentage = round(similarity * 100, 2)

        st.subheader(f"Match Score: {match_percentage}%")

        if match_percentage > 70:
            st.success("Great match! Your resume aligns well with the job.")
        elif match_percentage > 40:
            st.warning("Moderate match. Consider improving keywords.")
        else:
            st.error("Low match. You should customize your resume.")

        # Keyword analysis
        resume_words = set(resume_clean.split())
        jd_words = set(jd_clean.split())

        missing_keywords = jd_words - resume_words
        common_keywords = resume_words.intersection(jd_words)

        st.subheader("Common Keywords")
        st.write(list(common_keywords)[:20])

        st.subheader("Missing Keywords (Important to Add)")
        st.write(list(missing_keywords)[:20])

    else:
        st.warning("Please upload resume and paste job description.")
