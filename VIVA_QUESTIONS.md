# Viva Voce Questions and Answers

1. **What is your project?**  
   It is an NLP-based application that predicts the academic category of a student's question and recommends similar questions with stored answers.

2. **Why did you choose this topic?**  
   It addresses a practical student problem and demonstrates text classification and information retrieval using accessible tools.

3. **What is NLP?**  
   Natural Language Processing is a field of AI that helps computers process and analyze human language.

4. **What is TF-IDF?**  
   It is a text-vectorization method that weights terms using their frequency in a document and rarity across a collection.

5. **Why use TF-IDF?**  
   It is efficient, interpretable, and a strong baseline for text classification and retrieval.

6. **Why Logistic Regression?**  
   It is a fast supervised classifier that works well with sparse TF-IDF features and can provide class probabilities.

7. **What is cosine similarity?**  
   It measures the cosine of the angle between two vectors. Higher values generally indicate greater similarity for TF-IDF vectors.

8. **How are answers generated?**  
   They are retrieved from the CSV knowledge base; the project does not generate new answers.

9. **What is the difference between classification and retrieval?**  
   Classification assigns a category. Retrieval finds stored items similar to the user's question.

10. **What is a dataset?**  
    A structured collection of examples used to train, validate, test, or search a model.

11. **What is accuracy?**  
    The fraction of predictions that are correct.

12. **What are precision and recall?**  
    Precision measures how many predicted positives are correct; recall measures how many actual positives are found. In multiclass evaluation, the report computes class-wise and averaged metrics.

13. **What is an F1-score?**  
    The harmonic mean of precision and recall.

14. **What is a confusion matrix?**  
    A table comparing actual labels with predicted labels for each class.

15. **Why split data into train and test sets?**  
    To estimate performance on examples not used to fit the model.

16. **What are the limitations?**  
    A small dataset, word-overlap-based retrieval, and a fixed answer database limit the system.

17. **What happens if no answer matches?**  
    The app hides very low-similarity matches and asks the user to rephrase or expand the knowledge base.

18. **Why Streamlit?**  
    It creates a simple interactive web interface directly from Python without requiring separate frontend code.

19. **Can it answer every question?**  
    No. It can only recommend answers present in the database, and it may fail on unfamiliar wording.

20. **How would you improve it?**  
    Add verified data, evaluate with a separate query set, use semantic embeddings, and collect user feedback.

21. **What is overfitting?**  
    When a model fits training examples too closely and generalizes poorly to new data.

22. **Why is model confidence not a guarantee?**  
    Probabilities depend on the training data and model assumptions and can be miscalibrated, especially on unfamiliar inputs.

23. **What is the role of the CSV file?**  
    It stores question text, category, answer, and keywords. The application uses it as both labelled training examples and the retrieval knowledge base.

24. **What is the difference between semantic search and this project’s retrieval?**  
    This baseline mainly compares TF-IDF word/phrase vectors. Semantic search with embeddings can better match paraphrases with different wording.

25. **What is the main contribution?**  
    A small end-to-end prototype combining category prediction and answer retrieval in one student-friendly interface.
