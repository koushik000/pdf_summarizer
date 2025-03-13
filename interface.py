import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def download_nltk_data():
    """Downloads NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt_tab/english.pickle')
        nltk.data.find('corpora/stopwords')
        print("NLTK data already present.")  # Debug message
    except LookupError:
        with st.spinner("Downloading NLTK data... (This may take a minute)"):
            nltk.download('punkt_tab')
            nltk.download('stopwords')
        print("NLTK data downloaded successfully.")  # Debug message

# Download NLTK data (only runs once)
download_nltk_data()

@st.cache_data
def extract_text(pdf_file):
    """Extract text using PyMuPDF (fitz) with caching"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            text += doc[page_num].get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def clean_and_normalize_text(text, max_chars=100000):
    """Simplified text cleaning with size limit"""
    if not text:
        return ""

    # Limit text size for processing
    if len(text) > max_chars:
        text = text[:max_chars]

    # Essential cleaning operations
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

    return text

def is_valid_sentence(sentence):
    """Check if a string is likely a valid sentence (simplified)"""
    # Must have at least 5 characters
    if len(sentence) < 5:
        return False

    # Must have at least one letter
    if not any(c.isalpha() for c in sentence):
        return False

    # At least 3 words for a proper sentence
    words = sentence.split()
    if len(words) < 3:
        return False

    return True

def extract_sentences(text, max_sentences=1000):
    """Extract valid sentences from text with limit"""
    if not text.strip():
        return []

    # Split text into sentences
    sentences = sent_tokenize(text)

    # Filter out invalid sentences
    valid_sentences = [s for s in sentences if is_valid_sentence(s)]

    # Limit number of sentences for processing
    if len(valid_sentences) > max_sentences:
        valid_sentences = valid_sentences[:max_sentences]

    return valid_sentences

def get_top_sentences_tfidf(sentences, top_n=5, topic_focus=None):
    """Extract top sentences using TF-IDF and cosine similarity"""
    if not sentences:
        return []

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError as e:
        st.error(f"Error in TF-IDF processing: {e}")
        # Fallback to returning first few sentences
        return sentences[:min(top_n, len(sentences))]

    # If topic is specified, use it to extract focused sentences
    if topic_focus:
        topic_tfidf = vectorizer.transform([topic_focus])
        topic_similarity = cosine_similarity(topic_tfidf, tfidf_matrix)[0]

        # Get sentences most similar to topic
        top_idx = np.argsort(topic_similarity)[-top_n:]
        top_sentences = [sentences[i] for i in sorted(top_idx)]
        return top_sentences

    # Without topic, use sentence centrality (similarity to other sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Calculate sentence centrality
    centrality_scores = similarity_matrix.sum(axis=1)
    top_idx = np.argsort(centrality_scores)[-top_n:]
    top_sentences = [sentences[i] for i in sorted(top_idx)]

    return top_sentences

def format_summary(sentences, num_paragraphs=1):
    """Format sentences into paragraphs"""
    if not sentences:
        return "No valid content found to summarize."

    num_sentences = len(sentences)
    if num_paragraphs > num_sentences:
        num_paragraphs = num_sentences

    # Distribute sentences across paragraphs
    sentences_per_paragraph = num_sentences // num_paragraphs
    remainder = num_sentences % num_paragraphs

    paragraphs = []
    start_idx = 0

    for i in range(num_paragraphs):
        # Add extra sentence to early paragraphs if needed
        extra = 1 if i < remainder else 0
        end_idx = start_idx + sentences_per_paragraph + extra

        paragraph = " ".join(sentences[start_idx:end_idx])
        paragraphs.append(paragraph)

        start_idx = end_idx

    return "\n\n".join(paragraphs)

def summarize_pdf(pdf_file, topic_focus=None, num_paragraphs=1):
    """Main function to extract text and create summary with progress bar"""
    progress_bar = st.progress(0)

    # Step 1: Extract text
    progress_bar.progress(0.1)
    extracted_text = extract_text(pdf_file)

    # If no text extracted, show error
    if not extracted_text or len(extracted_text) < 100:
        progress_bar.progress(1.0)
        return {
            "success": False,
            "error": "Could not extract meaningful text from PDF."
        }

    # Step 2: Clean the text
    progress_bar.progress(0.3)
    cleaned_text = clean_and_normalize_text(extracted_text)

    # Step 3: Extract valid sentences
    progress_bar.progress(0.5)
    sentences = extract_sentences(cleaned_text)

    if not sentences:
        progress_bar.progress(1.0)
        return {
            "success": False,
            "error": "No valid sentences found after processing."
        }

    # Step 4: Get top sentences
    progress_bar.progress(0.7)
    num_sentences = max(5, num_paragraphs * 3)  # At least 3 sentences per paragraph
    top_sentences = get_top_sentences_tfidf(sentences, num_sentences, topic_focus)

    # Step 5: Format into paragraphs
    progress_bar.progress(0.9)
    summary = format_summary(top_sentences, num_paragraphs)

    progress_bar.progress(1.0)

    # Return results
    return {
        "success": True,
        "summary": summary,
        "sentence_count": len(sentences),
        "text_length": len(cleaned_text)
    }

# Streamlit UI
st.title("📄 PDF Summarizer")
st.caption("Upload a PDF to generate a summary")

# Main content
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # User options
    st.subheader("Summarization Options")
    topic_focus = st.text_input("Focus on specific topic (optional):",
                               placeholder="e.g., methodology, results, conclusions")

    num_paragraphs = st.slider("Number of paragraphs:", 1, 5, 2)

    if st.button("Summarize"):
        with st.spinner("Analyzing document..."):
            # Process the PDF
            result = summarize_pdf(uploaded_file, topic_focus, num_paragraphs)

            if result["success"]:
                st.subheader("Summary")
                st.write(result["summary"])
                st.caption(f"Created from {result['sentence_count']} sentences ({result['text_length']} characters)")
            else:
                st.error(result["error"])
else:
    st.info("Please upload a PDF document to begin.")