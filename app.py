from pathlib import Path
import re
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk import bigrams, pos_tag
from nltk.corpus import stopwords
from nltk.probability import FreqDist


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

# Corpus only loads ONCE when starting
@st.cache_resource

def corpus_preprocess(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        # Load and lowercase everything
        corpus_lower = f.read().lower()

    # 1. Make words into tokens, adds into list named "tokens"
    tokenizer = RegexpTokenizer(r'\b[a-z]+\b')
    tokens = tokenizer.tokenize(corpus_lower)

    # 2. Get frequency of each word (unigram)
    unigram_freq = Counter(tokens)
    
    # Only keep words that appears at least 2 times in case of potential typos
    vocab = set(word for word, count in unigram_freq.items() if count >= 2)
    
    # 3. Build Bigrams
    bigram_freq = FreqDist(bigrams(tokens))

    return unigram_freq, vocab, bigram_freq

# Put path to corpus file here
BASE = Path(__file__).parent
corpus = BASE / "medical_corpus.txt"
if not corpus.exists():
    corpus = BASE / "data" / "medical_corpus.txt"

# Preprocess corpus
unigram_freq, vocab, bigram_freq = corpus_preprocess(corpus)

# Stopwords - common words in english ("the", "and" , etc.)
stop_words = set(stopwords.words("english"))

# =========================
# Edit distance (Levenshtein)
# =========================
# Count how many character edits needed to transform word 1 to word 2
def edit_distance(w1, w2):
    m, n = len(w1), len(w2)

    # Initialize empty (m+1) × (n+1) matrix with zeros
    dp = [[0]*(n+1) for _ in range(m+1)]

    # Initialize matrix 
    for i in range(m+1):
        dp[i][0] = i        # cost / number of deletion from word 1 to empty string
    for j in range(n+1):
        dp[0][j] = j        # cost / number of insertion from empty string to word 2
    
    # Fill matrix using recurrence relation
    for i in range(1,m+1):
        for j in range(1,n+1):
            if w1[i-1] == w2[j-1]:          
                dp[i][j] = dp[i-1][j-1]     # no cost if characters match, cost = 0
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])    # Deletion, Insertion, Substitution cost (lowest edit)
    
    # Return Final Value, edit distance
    return dp[m][n]

# =========================
# Get candidate/suggested words
# =========================

def get_candidates(word, max_edit_dist=2):

    # Initialize empty list to store candidate/suggested words
    candidates = []

    # Count length of word used for comparison
    word_len = len(word)
    
    # Go through every word in corpus
    for w in vocab:

        # Filter, so that words that will have only 2 edit distace at most are obtained
        if abs(len(w) - word_len) <= max_edit_dist:
            
            # Skip same word that exist in corpus, will not be suggestion
            if w == word:
                continue
            
            # Get edit distance between entered word and word in corpus
            d = edit_distance(word, w)

            # Add suggested word into candidate list if edit distance within 2
            if d <= max_edit_dist:
                candidates.append((w, d))
                
    # Sort by: 
    # 1. Distance (lowest first)
    # 2. Frequency (thus using the minus sign)
    return sorted(candidates, key=lambda x: (x[1], -unigram_freq[x[0]]))

# =========================
# Real-word detection
# =========================
def is_real_word_error(word, prev_word, pos, threshold=100):
    """
    Returns True if 'word' is a real-word error based on neighborirng words
    'threshold' is how much more likely the neighbor must be.
    """
    # Skip if first word or if word is a very short like "in", "on" to reduce noise
    # 3 letter words often lead to false positives, thus skipped for stability
    if not prev_word or len(word) <= 3:
        return False
    
    # Skip the word if it is stop word
    if word in stop_words:
        return False

    # Check "Content Words"(Noun/Verb/Adjectives), dont check for prepositions
    content_tags = {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ"}
    if pos not in content_tags:
        return False

    # Get bigram frequency for current pair of words, if does not exist, return 0
    current_bigram_freq = bigram_freq.get((prev_word, word), 0)
    
    # Get candidates that are maximum 2 edit distance
    candidates = get_candidates(word, max_edit_dist=2)
    
    if not candidates:
        return False

    # Determine if suggested word is more likely to be correct compared to current word
    for cand_word, dist in candidates:
        #Get bigram frequency for suggested word and previous word, if does not exist, return 0
        candidate_bigram_freq = bigram_freq.get((prev_word, cand_word), 0)
        
        # If candidate is more frequent than the actual word, flag as real word error
        # Threshold is how likely is suggested word is correct compared to current word
        if candidate_bigram_freq > (current_bigram_freq * threshold):
            return True

    return False

# =========================
# Error Detection Logic
# =========================

def detect_errors(text):
    # Split sentence at whitespaces into tokens
    original_tokens = text.split() 
    
    # Pre-process tokens
    processed_tokens = [re.sub(r"[^a-z]", "", t.lower()) for t in original_tokens]

    # parts of speech (pos) tagging
    tags = pos_tag(processed_tokens)

    # Initialize list to store final results
    results = []

    # store previous word for real-word detection (bigram)
    prev_word = None

    for i, original_word in enumerate(original_tokens):

        # cleaned lowercase word
        w = processed_tokens[i]

        # part of speech tag
        pos = tags[i][1]

        # Create dictionary entry for word
        entry = {
            "index": i,
            "original_word": original_word,
            "cleaned_word": w,
            "error": False,
            "type": None,
            "candidates": []
        }

        # Skip empty strings (from conversion of number and symbols)
        if not w:
            results.append(entry)

            prev_word = None
            continue

        # Error type detection
        
        # Check if it's Non-word Error
        # If word does not exist in corpus, flag as non-word error
        if w not in vocab:
            entry["error"] = True
            entry["type"] = "Non-word"
            entry["candidates"] = get_candidates(w, max_edit_dist=2)

        # Check if it's Real-word Error
        # Check using previous word if current word fit context
        elif is_real_word_error(w, prev_word, pos):
            entry["error"] = True
            entry["type"] = "Real-word"
            entry["candidates"] = get_candidates(w, max_edit_dist=2)

        else:
            # If no errors are detected for currentword, current word is 
            # now used as "previous word" for next word (bigram)
            prev_word = w

        # Add into dictionary entry
        results.append(entry)

    return results

# =========================
# Highlight original text
# =========================
def highlight(results):

    # Empty list for storing text
    html_output = []

    # Loop through each entry of the sentence/text
    for entry in results:

        # Use original word 
        word = entry["original_word"]

        # If it is word error, highlight color according to error
        if entry["error"]:
            # Background color -  Red = Non-word errors, Blue = Real-word errors
            bg_color = "#FF4B4B" if entry["type"] == "Non-word" else "#1C83E1"
            span = f"<span style='background:{bg_color};color:white;padding:2px 6px;border-radius:4px'>{word}</span>"
            html_output.append(span)
        else:
            # If no error, return word without any formatting
            html_output.append(word)
    
    # Stitch words together with white space between
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

# Different tabs for spelling correction, dictionary and information
tab1, tab2, tab3 = st.tabs(["📝Spelling Correction", "📖 Dictionary", "ℹ️ About"])

# =========================
# Spell Check Tab
# =========================
with tab1:
    st.markdown("""
    ### 📘 How to Use
    1. Enter text or sentence of no more than 500 characters
    2. Click "Check Spelling" Button 
    3. Red highlights Non-word errors, blue highlights Real-word errors
    4. Select suggested words for sentence correction
    """)
    
    text = st.text_area(
        "Enter text for spelling correction:",
        "the aptient has diabtes ad inflamation form eating too much sugar as well as risk of heart attach",
        height=140,
        max_chars = 500     # Limit maximum characters allowed to 500
    )

    # Only runs code when button is pressed.
    if st.button("Check Spelling"):
        # Run error detection logic, results saved
        st.session_state.errors = detect_errors(text)
        # Real time correction of word when suggested word is selected
        st.session_state.corrected = {
            e["index"]: e["original_word"] for e in st.session_state.errors
        }

    # Only show highlighted errors and dropdown selection after button is clicked
    if st.session_state.errors:
        st.markdown("### Original text")
        # Pass results for errors to be highlighted
        st.markdown(highlight(st.session_state.errors), unsafe_allow_html=True)

        # split into left and right columns
        left, right = st.columns(2)

        # Non-word corrections
        with left:
            st.markdown("### Non-word Errors")

            #Loop through all non-word errors and create dropdown for selection
            for e in st.session_state.errors:
                if e["error"] and e["type"] == "Non-word":
                    # Show top 5 candidates with their edit distance
                    options = [f"{w} (edit distance ={d})" for w, d in e["candidates"][:5]]
                    # Create dropdown to select suggested word
                    choice = st.selectbox(f"Correction for: {e['original_word']}", options, key=f"nw_{e['index']}")
                    # Update and save the correction state
                    st.session_state.corrected[e["index"]] = choice.split()[0]

        with right:
            st.markdown("### Real-word Errors")
            for e in st.session_state.errors:

                # Loop through all real-word errors and create dropdown for selection
                if e["error"] and e["type"] == "Real-word":
                    # Keep original word in case of incorrect flag
                    original_word = f"{e['original_word']} (edit distance = 0)"
                    
                    # Show original word with top 5 candidates with their edit distance
                    options = [original_word] + [f"{w} (edit distance ={d})" for w, d in e["candidates"][:5]]
                    
                    # Create dropdown to select suggested word
                    choice = st.selectbox(f"Suggested word for: {e['original_word']}", options, key=f"rw_{e['index']}")
                    
                    # Update and save the correction state
                    st.session_state.corrected[e["index"]] = choice.split()[0]

        # Join back into original text sequence
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

    # Lets user input the string for searching
    query = st.text_input("Search word or substring").lower()
    
    # If text is entered, try to find the matching words
    if query:
        # Find matches
        matches = [w for w in vocab if query in w]
        # Create and display Table with 2 columns (word, frequency)
        search_results = pd.DataFrame({
            "word": matches,
            "unigram_frequency": [unigram_freq.get(w, 0) for w in matches]
        }).sort_values("unigram_frequency", ascending=False)                # Sort by descending freqency
        st.dataframe(search_results, width='stretch')

    st.divider() #Split searcher and browser

    st.subheader("🔤 Browse Word by First Alphabet")
    # Get all first alphabet of all words and sort from A to Z
    alphabet = sorted(list(set(w[0].upper() for w in vocab if w)))
    
    # Create dropdown selection for alphabet
    selected_alphabet = st.selectbox("Select first alphabet of word to browse", alphabet)

    if selected_alphabet:
        # Get all words that starts with selected alphabet
        matched_words = [w for w in vocab if w.startswith(selected_alphabet.lower())]
        
        # Create Data Frame for the words starting with selected alphabet and sort alphabetically
        alphabet_df = pd.DataFrame({
            "word": matched_words,
            "unigram_frequency": [unigram_freq.get(w, 0) for w in matched_words]
        }).sort_values("word")
        
        st.write(f"Showing {len(matched_words)} terms starting with '{selected_alphabet}'")
        st.dataframe(alphabet_df, width='stretch')

# =========================

# About Tab

# =========================

with tab3:

    st.markdown("""
    ## ℹ️ About This Application
    This spelling correction system uses **classical NLP techniques only**.

    - Minimum Edit Distance (find similar words for non-word error)
    - Unigram frequency model (ranking and prioritizing suggested words)
    - Bigram language model (real-word error detection)
    - Parts-Of-Speech tagging (filtering words and reduce false positives real-word error)

    """)

