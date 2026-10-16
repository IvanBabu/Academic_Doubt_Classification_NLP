# NLP-Based Academic Doubt Classification and Answer Recommendation System

## Overview
A beginner-friendly NLP course project. A student enters a question; the application predicts an academic category and retrieves similar stored questions with their answers.

## Features
- Academic doubt classification using TF-IDF + Logistic Regression
- Answer recommendation using TF-IDF + cosine similarity
- Confidence/probability display
- Top-N recommendations
- Editable CSV knowledge base
- Separate evaluation script with accuracy, precision, recall, F1-score, and confusion matrix
- Streamlit web interface

## Project structure
```
academic_doubt_nlp_project/
├── app.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── VIVA_QUESTIONS.md
├── PRESENTATION_OUTLINE.md
└── data/
    └── qa_dataset.csv
```

## Requirements
- Python 3.10 or 3.11 recommended
- Internet is required only to install Python packages the first time.
- No paid API key or GPU is required.

## Setup in VS Code
1. Extract the ZIP file.
2. Open the extracted `academic_doubt_nlp_project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   - Windows: `py -m venv .venv`
   - macOS/Linux: `python3 -m venv .venv`
5. Activate it:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows CMD: `.venv\Scripts\activate.bat`
   - macOS/Linux: `source .venv/bin/activate`
6. Install dependencies: `python -m pip install -r requirements.txt`
7. Start the app: `python -m streamlit run app.py`
8. Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## Evaluate the classifier
Run:
`python evaluate_model.py`

The script prints a held-out test accuracy, precision, recall, F1-score, and confusion matrix. Do not copy a guessed score into your report: use the score produced on your machine and explain the dataset's limitations.

## How to use the application
1. Type a complete academic question in the text box.
2. Select how many recommendations to show.
3. Click **Classify doubt and recommend answers**.
4. Review the predicted category, estimated probability, similar questions, answer text, and similarity scores.

## Add your own questions
Edit `data/qa_dataset.csv` in Excel or a text editor. Keep the exact columns:
`question,category,answer,keywords`
Use one row per question. Use consistent category names. Add examples for every category. Restart the app if necessary.

## How it works
1. TF-IDF converts question text to numeric sparse vectors.
2. Logistic Regression predicts one category from labelled questions.
3. Cosine similarity compares the new question with each knowledge-base question.
4. Results are sorted by similarity; low-scoring matches are hidden.

## Limitations
- The starter dataset is small and educational; results are not suitable for claiming production-level accuracy.
- Similarity is based on word/phrase overlap and may miss paraphrases.
- The system only returns answers present in its CSV knowledge base.
- Confidence is a model probability, not proof of correctness.
- Expand the dataset and manually review answers before formal deployment.

## Suggested extensions
- Add a feedback button (helpful/not helpful).
- Add user login and question history.
- Add PDF notes ingestion.
- Use sentence embeddings for stronger semantic retrieval.
- Add a teacher/admin page for maintaining Q&A entries.


## Expanded dataset and UI

The `data/qa_dataset.csv` file includes 144 sample question-answer records across 18 categories. The Streamlit interface uses custom CSS for a responsive dashboard-style layout. To run: `python -m streamlit run app.py`.
