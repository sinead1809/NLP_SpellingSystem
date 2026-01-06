import re
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk import bigrams, pos_tag
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from pathlib import Path

# =========================
# Ensure NLTK data (LOCAL + CLOUD SAFE)
# =========================

@st.cache_resource
def ensure_nltk_data():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")

ensure_nltk_data()

# =========================
# Load & preprocess corpus
# =========================

@st.cache_resource
def corpus_preprocess(corpus_file: Path):
    corpus_file = Path(corpus_file)

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_file}")

    with corpus_file.open("r", encoding="utf-8") as f:
        corpus_lower = f.read().lower()

    tokenizer = RegexpTokenizer(r"\b[a-z]+\b")
    tokens = tokenizer.tokenize(corpus_lower)

    unigram_freq = Counter(tokens)

    vocab_all = set(unigram_freq.keys())
    vocab_common = set(w for w, c in unigram_freq.items() if c >= 2)

    bigram_freq = FreqDist(bigrams(tokens))

    return unigram_freq, vocab_all, vocab_common, bigram_freq


# Robust path (works locally + cloud)
corpus_path = Path(__file__).parent / "data" / "medical_corpus.txt"

unigram_freq, vocab_all, vocab, bigram_freq = corpus_preprocess(corpus_path)
stop_words = set(stopwords.words("english"))

# =========================
# Edit Distance
# =========================

def edit_distance(w1, w2):
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i - 1] == w2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

# =========================
# Candidate Generation
# =========================

def get_candidates(word, max_edit_dist=2):
    candidates = []
    wl = len(word)

    for w in vocab:
        if w[0] != word[0]:
            continue
        if abs(len(w) - wl) <= max_edit_dist:
            d = edit_distance(word, w)
            if d <= max_edit_dist:
                candidates.append((w, d))

    return sorted(candidates, key=lambda x: (x[1], -unigram_freq[x[0]]))

# =========================
# Real-word Error Detection
# =========================

def is_real_word_error(word, prev_word, pos, threshold=100):
    if not prev_word or len(word) <= 3 or word in stop_words:
        return False

    content_tags = {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ"}
    if pos not in content_tags:
        return False

    current_freq = bigram_freq.get((prev_word, word), 0)
    for cand, _ in get_candidates(word):
        cand_freq = bigram_freq.get((prev_word, cand), 0)
        if cand_freq > current_freq * threshold:
            return True

    return False

# =========================
# Error Detection
# =========================

def detect_errors(text):
    original_tokens = text.split()
    cleaned_tokens = [re.sub(r"[^a-z]", "", t.lower()) for t in original_tokens]
    tags = pos_tag(cleaned_tokens)

    results = []
    prev_word = None

    for i, orig in enumerate(original_tokens):
        w = cleaned_tokens[i]
        pos = tags[i][1]

        entry = {
            "index": i,
            "original_word": orig,
            "error": False,
            "type": None,
            "candidates": []
        }

        if not w:
            results.append(entry)
            prev_word = None
            continue

        if w not in vocab_all:
            entry["error"] = True
            entry["type"] = "Non-word"
            entry["candidates"] = get_candidates(w)

        elif is_real_word_error(w, prev_word, pos):
            entry["error"] = True
            entry["type"] = "Real-word"
            entry["candidates"] = get_candidates(w)

        prev_word = w
        results.append(entry)

    return results

# =========================
# Highlighting
# =========================

def highlight(results):
    out = []
    for e in results:
        w = e["original_word"]
        if e["error"]:
            color = "#FF4B4B" if e["type"] == "Non-word" else "#1C83E1"
            out.append(f"<span style='background:{color};color:white;padding:3px 6px;border-radius:4px'>{w}</span>")
        else:
            out.append(w)
    return " ".join(out)

# =========================
# UI
# =========================

st.set_page_config(page_title="Spelling Correction System", layout="wide")
st.title("Spelling Correction System")

tab1, tab2, tab3 = st.tabs(["📝 Spelling Correction", "📖 Dictionary", "ℹ️ About"])

with tab1:
    text = st.text_area(
        "Enter text:",
        "the aptient has diabtes ad inflamation form eating too much sugar as well as risk of heart attach",
        height=140
    )

    if st.button("Check Spelling"):
        st.session_state.errors = detect_errors(text)

    if "errors" in st.session_state:
        st.markdown(highlight(st.session_state.errors), unsafe_allow_html=True)

with tab2:
    query = st.text_input("Search dictionary").lower()
    if query:
        matches = [w for w in vocab if query in w]
        df = pd.DataFrame({
            "word": matches,
            "frequency": [unigram_freq[w] for w in matches]
        }).sort_values("frequency", ascending=False)
        st.dataframe(df)

with tab3:
    st.markdown("""
    **Techniques used**
    - Minimum Edit Distance
    - Unigram frequency
    - Bigram language model
    - POS tagging
    """)

