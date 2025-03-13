# PDF Summarizer

A Streamlit web application that automatically extracts and summarizes content from PDF documents using natural language processing techniques.

## Features

- **PDF Text Extraction**: Efficiently extracts text content from uploaded PDF files
- **Smart Summarization**: Uses TF-IDF and cosine similarity to identify the most important sentences
- **Topic Focus**: Option to focus the summary on a specific topic or theme
- **Adjustable Length**: Control the summary length with paragraph settings
- **Progress Tracking**: Visual progress bar shows processing status

## Demo

[Try the live demo](https://summaryscribe.streamlit.app/) (Replace with your actual deployed URL)

![PDF Summarizer Screenshot]![image](https://github.com/user-attachments/assets/258da030-2f74-46a9-8033-12bda7a38607)
![image](https://github.com/user-attachments/assets/2e615436-1944-48f7-a8f3-17e900ad92bf)



## Installation

### Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pdf-summarizer.git
   cd pdf-summarizer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to http://localhost:8501

### Requirements

The following packages are required:
- streamlit
- PyMuPDF
- nltk
- scikit-learn
- numpy

## How It Works

1. **Text Extraction**: The application extracts text from the uploaded PDF using PyMuPDF.
2. **Text Cleaning**: The extracted text is cleaned to remove artifacts and normalize content.
3. **Sentence Extraction**: The text is split into individual sentences.
4. **Importance Scoring**: TF-IDF vectorization and cosine similarity are used to identify key sentences.
5. **Summary Generation**: Top-scoring sentences are combined into a coherent summary.

## Deployment

This application can be deployed on Streamlit Community Cloud:

1. Fork this repository to your GitHub account
2. Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Create a new app and select your forked repository
4. Deploy and share the generated URL

## Limitations

- Processing large PDFs (100+ pages) may take longer
- PDFs with complex formatting or primarily image-based content may not extract well
- The summarizer works best with well-structured academic or business documents

## Future Improvements

- Add support for OCR to handle scanned documents
- Implement additional summarization algorithms
- Add multi-language support
- Enable batch processing of multiple PDFs

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses [Streamlit](https://streamlit.io/) for the web interface
- PDF processing is handled by [PyMuPDF](https://pymupdf.readthedocs.io/)
- Text analysis uses [NLTK](https://www.nltk.org/) and [scikit-learn](https://scikit-learn.org/)
