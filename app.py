import re
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, pos_tag
from nltk.corpus import stopwords
from nltk.probability import FreqDist


# =========================
# Ensure NLTK data (LOCAL + CLOUD SAFE)
# =========================
@st.cache_resource
def ensure_nltk_data():
    nltk.download("punkt")
    nltk.download("stopwords")
    # NLTK versions differ; download both tagger names safely
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

    # Tokenise words (letters only)
    tokenizer = RegexpTokenizer(r"\b[a-z]+\b")
    tokens = tokenizer.tokenize(corpus_lower)

    # Unigram frequency
    unigram_freq = Counter(tokens)

    # vocab_all: all observed words (for "known word?" checks + dictionary)
    vocab_all = set(unigram_freq.keys())

    # vocab_common: only frequent words (for candidate generation)
    vocab_common = set(w for w, c in unigram_freq.items() if c >= 2)

    # Bigram frequency
    bigram_freq = FreqDist(bigrams(tokens))

    return unigram_freq, vocab_all, vocab_common, bigram_freq


# -------------------------
# Corpus path (works in GitHub/Streamlit Cloud)
# -------------------------
BASE = Path(__file__).parent
corpus_path = BASE / "medical_corpus.txt"
if not corpus_path.exists():
    corpus_path = BASE / "data" / "medical_corpus.txt"

unigram_freq, vocab_all, vocab_common, bigram_freq = corpus_preprocess(corpus_path)

# Stopwords
stop_words = set(stopwords.words("english"))


# =========================
# Edit distance (Levenshtein)
# =========================
def edit_distance(w1: str, w2: str) -> int:
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
# Get candidate/suggested words
# =========================
def get_candidates(word: str, max_edit_dist: int = 2):
    candidates = []
    word_len = len(word)

    for w in vocab_common:
        # cheap length filter
        if abs(len(w) - word_len) > max_edit_dist:
            continue
        if w == word:
            continue

        d = edit_distance(word, w)
        if d <= max_edit_dist:
            candidates.append((w, d))

    # sort: distance asc, frequency desc
    return sorted(candidates, key=lambda x: (x[1], -unigram_freq.get(x[0], 0)))


# =========================
# Real-word detection (Bigram + POS + Stopwords)
# =========================
def is_real_word_error(word: str, prev_word: str | None, pos: str, threshold: int = 100) -> bool:
    """
    Flags a real-word error if a close candidate forms a far more frequent bigram
    with the previous word.
    """
    if not prev_word or len(word) <= 3:
        return False
    if word in stop_words:
        return False

    content_tags = {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ"}
    if pos not in content_tags:
        return False

    current_bigram_freq = bigram_freq.get((prev_word, word), 0)
    candidates = get_candidates(word, max_edit_dist=2)

    if not candidates:
        return False

    for cand_word, _dist in candidates:
        candidate_bigram_freq = bigram_freq.get((prev_word, cand_word), 0)
        if candidate_bigram_freq > (current_bigram_freq * threshold):
            return True

    return False


# =========================
# Error Detection Logic
# =========================
def detect_errors(text: str):
    original_tokens = text.split()
    processed_tokens = [re.sub(r"[^a-z]", "", t.lower()) for t in original_tokens]

    tags = pos_tag(processed_tokens)

    results = []
    prev_word = None

    for i, original_word in enumerate(original_tokens):
        w = processed_tokens[i]
        pos = tags[i][1]

        entry = {
            "index": i,
            "original_word": original_word,
            "cleaned_word": w,
            "error": False,
            "type": None,
            "candidates": []
        }

        if not w:
            results.append(entry)
            prev_word = None
            continue

        # Non-word: not seen in corpus at all
        if w not in vocab_all:
            entry["error"] = True
            entry["type"] = "Non-word"
            entry["candidates"] = get_candidates(w, max_edit_dist=2)

        # Real-word: seen but context unlikely
        elif is_real_word_error(w, prev_word, pos):
            entry["error"] = True
            entry["type"] = "Real-word"
            entry["candidates"] = get_candidates(w, max_edit_dist=2)

        results.append(entry)

        # IMPORTANT: always update context for next word
        prev_word = w

    return results


# =========================
# Highlight original text
# =========================
def highlight(results):
    html_output = []

    for entry in results:
        word = entry["original_word"]

        if entry["error"]:
            bg_color = "#FF4B4B" if entry["type"] == "Non-word" else "#1C83E1"
            span = (
                f"<span style='background:{bg_color};color:white;"
                f"padding:2px 6px;border-radius:4px'>{word}</span>"
            )
            html_output.append(span)
        else:
            html_output.append(word)

    return " ".join(html_output)


# =========================
# Session state
# =========================
if "errors" not in st.session_state:
    st.session_state.errors = None
if "corrected" not in st.session_state:
    st.session_state.corrected = {}


# =========================
# UI Setup
# =========================
st.set_page_config(layout="wide", page_title="Spelling Correction System")
st.title("Spelling Correction System")

tab1, tab2, tab3 = st.tabs(["📝 Spelling Correction", "📖 Dictionary", "ℹ️ About"])


# =========================
# Spell Check Tab
# =========================
with tab1:
    st.markdown("""
### 📘 How to Use
1. Enter text (maximum **500 characters**)
2. Click **Check Spelling**
3. **Red** = Non-word errors, **Blue** = Real-word errors
4. Choose suggested replacements to generate a corrected sentence
""")

    text = st.text_area(
        "Enter text for spelling correction:",
        "the aptient has diabtes ad inflamation form eating too much sugar as well as risk of heart attach",
        height=140,
        max_chars=500
    )

    if st.button("Check Spelling"):
        st.session_state.errors = detect_errors(text)
        # initialise correction map with original words
        st.session_state.corrected = {e["index"]: e["original_word"] for e in st.session_state.errors}

    if st.session_state.errors:
        st.markdown("### Original text")
        st.markdown(highlight(st.session_state.errors), unsafe_allow_html=True)

        left, right = st.columns(2)

        # -------- Non-word corrections --------
        with left:
            st.markdown("### Non-word Errors")
            found = False

            for e in st.session_state.errors:
                if e["error"] and e["type"] == "Non-word":
                    found = True
                    options = [w for w, _d in e["candidates"][:5]]  # HIDE edit distance in UI
                    if not options:
                        options = [e["original_word"]]

                    choice = st.selectbox(
                        f"Correction for: {e['original_word']}",
                        options,
                        key=f"nw_{e['index']}"
                    )
                    st.session_state.corrected[e["index"]] = choice

            if not found:
                st.caption("No non-word errors detected.")

        # -------- Real-word corrections --------
        with right:
            st.markdown("### Real-word Errors")
            found = False

            for e in st.session_state.errors:
                if e["error"] and e["type"] == "Real-word":
                    found = True
                    options = [e["original_word"]] + [w for w, _d in e["candidates"][:5]]  # include original word
                    # remove duplicates while preserving order
                    seen = set()
                    options = [x for x in options if not (x in seen or seen.add(x))]

                    choice = st.selectbox(
                        f"Suggested word for: {e['original_word']}",
                        options,
                        key=f"rw_{e['index']}"
                    )
                    st.session_state.corrected[e["index"]] = choice

            if not found:
                st.caption("No real-word errors detected.")

        # Join back into original sequence
        final_sentence = " ".join(
            st.session_state.corrected[i] for i in range(len(st.session_state.errors))
        )

        st.subheader("Corrected Sentence")
        st.text_area("Final Output", final_sentence, height=120)


# =========================
# Dictionary Tab
# =========================
with tab2:
    st.subheader("🔍 Dictionary Search")

    query = st.text_input("Search word or substring").lower().strip()

    if query:
        matches = [w for w in vocab_all if query in w]
        search_results = pd.DataFrame({
            "word": matches,
            "unigram_frequency": [unigram_freq.get(w, 0) for w in matches]
        }).sort_values("unigram_frequency", ascending=False)

        st.dataframe(search_results, use_container_width=True)

    st.divider()

    st.subheader("🔤 Browse words by first letter")

    alphabet = sorted(list(set(w[0].upper() for w in vocab_all if w)))
    selected_alphabet = st.selectbox("Select first letter", ["All"] + alphabet)

    if selected_alphabet != "All":
        matched_words = [w for w in vocab_all if w.startswith(selected_alphabet.lower())]
    else:
        matched_words = list(vocab_all)

    alphabet_df = pd.DataFrame({
        "word": sorted(matched_words),
        "unigram_frequency": [unigram_freq.get(w, 0) for w in sorted(matched_words)]
    })

    st.write(f"Showing {len(matched_words)} terms")
    st.dataframe(alphabet_df, use_container_width=True)


# =========================
# About Tab
# =========================
with tab3:
    st.markdown("""
## ℹ️ About This Application

This spelling correction system uses **classical NLP techniques**:

- **Minimum Edit Distance (Levenshtein)**: generates candidate corrections.
- **Unigram frequency model**: ranks and prioritises suggested words.
- **Bigram language model**: detects real-word errors using context.
- **POS tagging + stopword filtering**: reduces false positives in real-word detection.

The interface is implemented with **Streamlit** to allow interactive testing and evaluation.
""")
