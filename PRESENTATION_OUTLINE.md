# Presentation Outline (10 slides)

## Slide 1 — Title
NLP-Based Academic Doubt Classification and Answer Recommendation System  
Student name, register number, department, guide, institution.

## Slide 2 — Problem Statement
- Students search several notes and resources to answer academic doubts.
- Manual searching takes time.
- A question may not be routed to the correct subject resource.

## Slide 3 — Objectives
- Classify academic questions.
- Retrieve similar stored questions.
- Recommend relevant answers.
- Provide a simple interface.

## Slide 4 — Existing vs Proposed System
Existing: manual search through notes and websites.  
Proposed: category prediction + ranked FAQ recommendations from a local knowledge base.

## Slide 5 — Technology Stack
Python, pandas, scikit-learn, TF-IDF, Logistic Regression, cosine similarity, Streamlit, CSV.

## Slide 6 — System Architecture
Question input → TF-IDF → category classifier → cosine-similarity retrieval → ranked answers → Streamlit output.

## Slide 7 — Working Example
Input: “What is the difference between TCP and UDP?”  
Show predicted category, estimated class probability, top matches, and answer.

## Slide 8 — Evaluation
Run `python evaluate_model.py` and show actual accuracy, precision, recall, F1-score, and confusion matrix. Explain that a small dataset limits conclusions.

## Slide 9 — Advantages and Limitations
Advantages: simple, low-cost, transparent, editable database.  
Limitations: small dataset, lexical matching, fixed answers.

## Slide 10 — Conclusion and Future Scope
The prototype demonstrates NLP classification and retrieval. Future work: larger verified dataset, semantic embeddings, PDF notes search, feedback collection.

**Presentation tip:** Demonstrate the app live with one question from each of two or three categories. Use actual model evaluation output rather than claiming an unmeasured accuracy.
