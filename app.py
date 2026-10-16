from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="StudyMate | Academic Doubt Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "qa_dataset.csv"

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17233c;
        --muted: #64748b;
        --primary: #5b5ce2;
        --teal: #0f9f9a;
        --line: #e6eaf2;
        --surface: #ffffff;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 5% 0%, rgba(91,92,226,.10), transparent 28rem),
            radial-gradient(circle at 95% 4%, rgba(15,159,154,.09), transparent 25rem),
            #f6f8fc;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(246,248,252,.88); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171c3a 0%, #202957 100%);
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #f4f6ff; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1240px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: -.025em; }
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 26px;
        padding: 30px 34px;
        margin-bottom: 24px;
        color: white;
        background: linear-gradient(120deg, #272b65 0%, #5757d9 57%, #168e9b 120%);
        box-shadow: 0 18px 42px rgba(46,54,126,.18);
    }
    .hero:after {
        content: "✦";
        position: absolute;
        right: 42px; top: -36px;
        font-size: 170px;
        line-height: 1;
        color: rgba(255,255,255,.09);
        transform: rotate(12deg);
    }
    .hero-kicker {
        display: inline-block; padding: 6px 11px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.28);
        background: rgba(255,255,255,.12); font-size: .78rem;
        font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
    }
    .hero h1 { color: #fff !important; font-size: clamp(1.8rem, 3vw, 2.65rem); margin: 13px 0 8px; line-height: 1.14; }
    .hero p { color: rgba(255,255,255,.86); font-size: 1.02rem; max-width: 720px; margin: 0; }
    .hero-foot { margin-top: 19px; color: rgba(255,255,255,.8); font-size: .84rem; }
    .section-title { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.25rem; color: #17233c; margin: 22px 0 12px; }
    .stat-card {
        background: rgba(255,255,255,.92); border: 1px solid var(--line);
        border-radius: 18px; padding: 18px 20px; min-height: 112px;
        box-shadow: 0 6px 22px rgba(28,43,76,.045);
    }
    .stat-label { color: #718096; font-size: .84rem; font-weight: 600; }
    .stat-value { color: #202b4a; font-family: 'Manrope', sans-serif; font-size: 1.8rem; font-weight: 800; margin-top: 5px; }
    .stat-note { color: #8792a7; font-size: .78rem; margin-top: 1px; }
    .panel {
        background: rgba(255,255,255,.96); border: 1px solid var(--line);
        border-radius: 20px; padding: 22px 24px;
        box-shadow: 0 8px 26px rgba(28,43,76,.045); margin-bottom: 14px;
    }
    .result-top { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    .category-pill {
        display:inline-block; background:#eeefff; color:#4c4dd0;
        border:1px solid #dcdcff; border-radius:999px; padding:7px 12px;
        font-size:.84rem; font-weight:700;
    }
    .score-pill {
        display:inline-block; background:#e6faf5; color:#087d70;
        border:1px solid #c6f1e7; border-radius:999px; padding:6px 10px;
        font-size:.8rem; font-weight:700;
    }
    .answer-label { color:#7b879b; text-transform:uppercase; letter-spacing:.07em; font-size:.73rem; font-weight:800; margin:15px 0 5px; }
    .answer-text { color:#34415a; font-size:1rem; line-height:1.72; }
    .tiny-note { color:#8190a7; font-size:.82rem; }
    div.stButton > button[kind="primary"] {
        border: 0; border-radius: 13px; min-height: 3.15rem;
        background: linear-gradient(100deg, #5657dc, #7873f0);
        color: white; font-weight: 700; box-shadow: 0 8px 18px rgba(86,87,220,.2);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px); box-shadow: 0 12px 24px rgba(86,87,220,.27);
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stSlider {
        border-radius: 12px !important;
        min-height: 3.2rem;
    }
    /* Make the editable field and its hint readable in light and dark themes. */
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #17233c !important;
        -webkit-text-fill-color: #17233c !important;
        border: 1px solid #dbe2ef !important;
        padding-left: 16px !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #69788f !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #69788f !important;
    }
    [data-testid="stMetric"] {
        background: white; padding: 15px 18px; border-radius: 16px;
        border: 1px solid var(--line);
    }
    .footer {
        text-align:center; color:#94a0b3; font-size:.8rem; padding: 24px 0 5px;
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 23px 21px; border-radius: 20px; }
        .hero:after { right: 5px; font-size: 110px; }
        .panel { padding: 17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Ensure data/qa_dataset.csv exists."
        )
    df = pd.read_csv(DATA_PATH).fillna("")
    required = {"question", "category", "answer", "keywords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "") & (df["category"] != "")]
    return df.reset_index(drop=True)

@st.cache_resource
def build_models():
    df = load_data()
    # Include keywords in both classification and retrieval representations.
    text = (df["question"] + " " + df["keywords"]).str.strip().tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    question_matrix = vectorizer.fit_transform(text)
    classifier = LogisticRegression(
        max_iter=1500, class_weight="balanced", random_state=42
    )
    classifier.fit(question_matrix, df["category"])
    return vectorizer, question_matrix, classifier

df = load_data()
vectorizer, question_matrix, classifier = build_models()
categories = sorted(df["category"].unique())
category_icons = {
    "Mathematics": "➗", "Science": "🔬", "Physics": "⚡", "Chemistry": "🧪",
    "Biology": "🧬", "History": "🏛️", "Geography": "🌍", "Social Studies": "🤝",
    "English": "📚", "Economics": "📈", "Civics": "⚖️", "General Knowledge": "💡",
    "Environmental Studies": "🌱", "Programming": "💻", "AI/ML": "🤖",
    "DBMS": "🗄️", "Networking": "🌐", "NLP": "📝",
}

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎓 StudyMate")
    st.markdown("Your academic question companion")
    st.markdown("---")
    st.markdown("### 📊 Knowledge base")
    st.metric("Questions & answers", len(df))
    st.metric("Subject categories", df["category"].nunique())
    st.markdown("---")
    st.markdown("### 🧭 Explore subjects")
    for category in categories:
        st.markdown(f"{category_icons.get(category, '📘')}  {category}")
    st.markdown("---")
    st.markdown("### 💬 How it works")
    st.markdown(
        "1. Enter your doubt\\n"
        "2. Predict the subject with NLP\\n"
        "3. Retrieve similar stored questions\\n"
        "4. Review the recommended answer"
    )
    st.caption("This prototype retrieves answers from the CSV; it does not generate new facts.")

# ---------- Hero ----------
st.markdown(
    f"""
    <div class="hero">
      <span class="hero-kicker">NLP • Machine Learning • Information Retrieval</span>
      <h1>Learn smarter.<br>Find answers faster.</h1>
      <p>Ask an academic question and StudyMate will predict its subject and search a curated question-and-answer library for useful matches.</p>
      <div class="hero-foot">✨ {len(df)} sample Q&A records &nbsp; • &nbsp; 📚 {df['category'].nunique()} subject areas &nbsp; • &nbsp; ⚙️ TF-IDF + Logistic Regression</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Overview cards
m1, m2, m3 = st.columns(3)
for col, label, value, note in [
    (m1, "Knowledge base", f"{len(df)}", "Curated Q&A examples"),
    (m2, "Subject areas", f"{df['category'].nunique()}", "Academic categories"),
    (m3, "Search method", "TF-IDF", "Text similarity retrieval"),
]:
    with col:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div><div class="stat-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title">✍️ Ask your academic doubt</div>', unsafe_allow_html=True)
with st.container():
    question = st.text_input(
        "Your question",
        label_visibility="collapsed",
        key="question_input",
        placeholder="Type your academic question here…",
        help="Enter a complete question for better classification and retrieval.",
    )
    top_k = 5
    submitted = st.button(
        "🔎  Classify doubt & find answers",
        type="primary",
        use_container_width=True,
    )

if submitted:
    cleaned = question.strip()
    if len(cleaned) < 3:
        st.warning("Please enter a question with a few meaningful words.")
    else:
        input_vec = vectorizer.transform([cleaned])
        probabilities = classifier.predict_proba(input_vec)[0]
        best_idx = int(probabilities.argmax())
        predicted_category = str(classifier.classes_[best_idx])
        confidence = float(probabilities[best_idx])

        st.markdown('<div class="section-title">📌 Your doubt analysis</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1.35, 1])
        with r1:
            st.markdown(
                f'<div class="panel"><div class="stat-label">PREDICTED SUBJECT</div>'
                f'<div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;color:#292d68;margin:7px 0 12px;">'
                f'{category_icons.get(predicted_category, "📘")} {predicted_category}</div>'
                f'<span class="category-pill">Most likely category</span>'
                f'<p class="tiny-note" style="margin-top:14px;">The model estimates the subject from the wording of your question.</p></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="panel"><div class="stat-label">MODEL CONFIDENCE</div>'
                f'<div style="font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;color:#087d70;margin:7px 0 4px;">{confidence:.1%}</div>'
                f'<div class="tiny-note">Estimated probability, not a guarantee</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(confidence, 0.0), 1.0))
        if confidence < 0.35:
            st.info("The model is uncertain about the subject. Try adding a subject-specific term or reviewing the top category probabilities.")

        # Similarity-based retrieval across the full knowledge base.
        similarities = cosine_similarity(input_vec, question_matrix).ravel()
        ranked_indices = similarities.argsort()[::-1]
        # A small threshold removes only completely unrelated matches; include the best hit when there is any lexical overlap.
        eligible = [int(i) for i in ranked_indices if float(similarities[i]) > 0.015]
        st.markdown('<div class="section-title">✨ Recommended answers</div>', unsafe_allow_html=True)
        if not eligible:
            st.warning(
                "I couldn't find meaningful word overlap with the current knowledge base. "
                "Try a different wording or add a relevant Q&A entry to data/qa_dataset.csv."
            )
        else:
            if float(similarities[eligible[0]]) < 0.08:
                st.info("These are the closest available matches, but relevance may be limited. Review them before relying on the answer.")
            for rank, idx in enumerate(eligible[:int(top_k)], start=1):
                item = df.iloc[idx]
                score = float(similarities[idx])
                st.markdown(
                    f'<div class="panel"><div class="result-top">'
                    f'<span class="category-pill">{category_icons.get(item["category"], "📘")} {item["category"]}</span>'
                    f'<span class="score-pill">{score:.0%} text similarity</span></div>'
                    f'<div class="answer-label">MATCH {rank} · QUESTION</div>'
                    f'<div style="font-family:Manrope,sans-serif;font-weight:800;font-size:1.1rem;color:#202b4a;">{item["question"]}</div>'
                    f'<div class="answer-label">RECOMMENDED ANSWER</div>'
                    f'<div class="answer-text">{item["answer"]}</div></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("📊 See category probability breakdown"):
            prob_df = pd.DataFrame({
                "Category": classifier.classes_,
                "Probability": probabilities,
            }).sort_values("Probability", ascending=False)
            st.dataframe(
                prob_df.style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

st.markdown(
    '<div class="footer">StudyMate · Academic Doubt Classification & Answer Recommendation · '
    'Course project prototype — verify answers before academic use.</div>',
    unsafe_allow_html=True,
)
