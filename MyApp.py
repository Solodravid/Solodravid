import streamlit as st # type: ignore
import pandas as pd
import time
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer  # Corrected import
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load External CSS
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar Navigation
st.sidebar.markdown("""
    <div class='sidebar-container'>
        <h2>📌 Navigation</h2>
        <ul>
            <li>📂 Upload resumes (PDF)</li>
            <li>📝 Enter job description</li>
            <li>🎯 Click 'Rank Resumes'</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Title with Animation
st.markdown("<div class='title-container'><h1>🚀 AI Resume Screening & Ranking</h1></div>", unsafe_allow_html=True)

# Job Description Input
st.markdown("<div class='input-container'><h3>✍️ Enter Your Job Description</h3></div>", unsafe_allow_html=True)
job_description = st.text_area("", height=150, placeholder="Paste the job description here...")

# Resume Upload
st.markdown("<div class='input-container'><h3>📄 Upload Resumes (PDF Only)</h3></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["pdf"], label_visibility="collapsed")

# Extract Text from PDFs
def extract_text(file):
    pdf_reader = PdfReader(file)
    return " ".join(page.extract_text() or "" for page in pdf_reader.pages)

# Extract Keywords from Resume
def extract_keywords(text, top_n=5):
    words = text.lower().split()
    common_words = Counter(words).most_common(top_n)
    return [word[0] for word in common_words]

# Rank Resumes Based on Job Description
def rank_resumes(job_desc, resumes):
    if not job_desc or not resumes:
        return []
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_desc] + resumes)
    similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]

    scores = [score * 100 for score in similarities]
    ranked_resumes = list(zip(resumes, scores))
    
    return sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

# Rank Button
if st.button("🎯 Rank Resumes", key="rank_button"):
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    elif not job_description.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        with st.spinner("🔍 Processing..."):
            time.sleep(2)  # Simulated Loading Effect
            resume_texts = [extract_text(f) for f in uploaded_files if f is not None]
            ranked = rank_resumes(job_description, resume_texts)

        # Display Results
        st.markdown("<div class='results-container'><h2>🏆 Ranked Resumes</h2></div>", unsafe_allow_html=True)
        ranked_data = []

        for i, (resume, score) in enumerate(ranked):
            color_class = "high-score" if score >= 60 else "medium-score" if score >= 35 else "low-score"
            top_keywords = extract_keywords(resume)

            # Display resume ranking with card effect
            st.markdown(
                f"""
                <div class="resume-card {color_class}">
                    <h3>📜 Resume {i+1}</h3>
                    <p>🎯 Score: {score:.2f}/100</p>
                    <p>🔑 Keywords: {', '.join(top_keywords)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            ranked_data.append({"Resume": f"Resume {i+1}", "Score": score, "Keywords": ", ".join(top_keywords)})

        # Download Option
        if ranked_data:
            df = pd.DataFrame(ranked_data)
            st.download_button(
                label="📥 Download Results",
                data=df.to_csv(index=False),
                file_name="ranked_resumes.csv",
                mime="text/csv",
                key="download_button"
            )