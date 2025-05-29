# --- Imports ---

from flask_cors import CORS
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('maxent_ne_chunker_tab', quiet=True)
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
from collections import Counter
from langdetect import detect as lang_detect
from stopwordsiso import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score
from flask import Flask, request, jsonify

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

# --- NLTK Resource Check ---
REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words"
}

logging.info("Checking NLTK resources...")
for download_key, path_to_find in REQUIRED_NLTK_RESOURCES.items():
    try:
        nltk.data.find(path_to_find)
        logging.info(f"NLTK resource '{download_key}' found.")
    except LookupError:
        logging.warning(f"NLTK resource '{download_key}' not found. Downloading...")
        try:
            nltk.download(download_key, quiet=True)
            nltk.data.find(path_to_find)
            logging.info(f"NLTK resource '{download_key}' downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download NLTK resource '{download_key}': {e}")

# --- Global Settings for Authorship Attribution ---
_EN_SW = set()
_JA_SW = set()
try:
    _EN_SW = stopwords("en")
    _JA_SW = stopwords("ja")
    logging.info("English and Japanese stopwords loaded.")
except Exception as e:
    logging.error(f"Failed to load stopwords: {e}")
_ALL_SW = sorted(list(_EN_SW.union(_JA_SW)))

_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# --- Flask App Setup ---
app = Flask(__name__)
allowed_origins = ["http://localhost:3000"]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# === Text processing for KWIC Search ===
def get_text_from_url_for_kwic(url):
    logging.info(f"Fetching URL for KWIC search: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()):
                logging.warning(f"Non-HTML content: {content_type}")
                raise ValueError("URL is not an HTML page.")
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript', 'nav', 'footer', 'aside', 'form', 'input', 'button', 'img', 'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched URL: {url}")
        return text
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error: {e.reason}")
        raise ValueError(f"Could not access URL: {e.reason}")
    except TimeoutError:
        logging.error(f"Timeout fetching {url}")
        raise ValueError("URL fetch timed out.")
    except Exception as e:
        logging.error(f"Unexpected error in get_text_from_url_for_kwic: {e}\n{traceback.format_exc()}")
        raise ValueError("Unexpected error while fetching URL.")

def clean_text_for_kwic(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Authorship Attribution Helpers ===
def fetch_wikipedia_text_for_authorship(url: str) -> str:
    logging.info(f"Fetching Wikipedia text for authorship: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            html = response.read()
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()):
                raise ValueError("URL is not an HTML page.")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "sup", "table", "nav", "footer", "aside", "header", "form", "figure", "figcaption", "link", "meta", "input", "button", "img", "audio", "video", "iframe", "object", "embed", "svg", "canvas", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ")
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except (URLError, HTTPError, TimeoutError, ValueError) as e:
        logging.error(f"Error fetching URL: {e}")
        raise ValueError(f"Could not fetch URL: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise ValueError("Unexpected error during URL fetch.")

def mixed_sentence_tokenize_for_authorship(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def tokenize_mixed_for_authorship(text: str):
    _PUNCT_SKIP = {".", ",", "(", ")", "'"}
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        logging.error(f"Tokenization failed: {e}")
        tokens = text.split()
    tokens = [t for t in tokens if t not in _PUNCT_SKIP]
    return tokens

def build_sentence_dataset_for_authorship(text: str, author_label: str, min_len: int = 30):
    sentences = mixed_sentence_tokenize_for_authorship(text)
    filtered = [s for s in sentences if len(s) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels

# === API Endpoints ===
@app.route('/api/search', methods=['POST'])
def kwic_search():
    data = request.json
    url = data.get('url', '').strip()
    query_input = data.get('query', '').strip()
    search_type = data.get('type', '').strip()

    logging.info(f"KWIC search request: URL={url}, Query='{query_input}', Type={search_type}")

    # Input validation
    if not url:
        return jsonify({"error": "Please provide a Wikipedia URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https'] or not parsed_url.hostname:
            return jsonify({"error": "Invalid URL scheme or hostname."}), 400
        hostname_lower = parsed_url.hostname.lower()
        if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
            return jsonify({"error": "Only Wikipedia URLs are supported."}), 400
    except Exception as e:
        logging.error(f"URL parsing error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Invalid URL format."}), 400

    if not query_input:
        return jsonify({"error": "Please provide a search query."}), 400

    target_words = query_input.split()
    if not 1 <= len(target_words) <= 5:  # Adjusted to match App.tsx max 5 words
        return jsonify({"error": "Enter 1 to 5 words for the target phrase."}), 400

    # Map search_type to internal mode
    type_map = {'token': '1', 'pos': '3', 'entity': '2'}  # ←ここを修正
    if search_type not in type_map:
        return jsonify({"error": "Invalid mode. Please choose 'token', 'pos', or 'entity'."}), 400
    mode = type_map[search_type]

    # Fetch and process text
    try:
        raw_text = get_text_from_url_for_kwic(url)
        text_cleaned = clean_text_for_kwic(raw_text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Text processing error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Server error during text processing."}), 500

    if not text_cleaned:
        return jsonify({"results": [], "error": "No searchable text extracted."}), 200

    # Detect language
    try:
        lang = lang_detect(text_cleaned) if text_cleaned else "en"
    except Exception:
        lang = "en"
    logging.info(f"Detected language: {lang}")

    # Tokenize text
    try:
        tokens_original = tokenize_mixed_for_authorship(text_cleaned)
        tokens_lower = [t.lower() for t in tokens_original]
    except Exception as e:
        logging.error(f"Tokenization failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Text tokenization failed."}), 500

    if not tokens_original:
        return jsonify({"results": [], "error": "No tokens extracted."}), 200

    results = []
    context_window = 5
    target_token_list = [w.lower() for w in target_words]
    num_target_tokens = len(target_token_list)

    if mode in ['1', '2']:
        following_words = []
        contexts = []
        for i in range(len(tokens_lower) - num_target_tokens + 1):
            if tokens_lower[i:i + num_target_tokens] == target_token_list:
                next_idx = i + num_target_tokens
                if next_idx < len(tokens_lower):
                    next_token = tokens_original[next_idx]
                    context_words = tokens_original[max(0, i - context_window):next_idx + context_window + 1]
                    matched_start = i - max(0, i - context_window)
                    matched_end = matched_start + num_target_tokens
                    following_words.append(next_token)
                    contexts.append({
                        "context_words": context_words,
                        "matched_start": matched_start,
                        "matched_end": matched_end
                    })

        if not following_words:
            return jsonify({"results": [], "error": f"No matches found for '{query_input}'."}), 200

        token_counts = Counter(following_words)
        sorted_tokens = [token for token, _ in token_counts.most_common()]

        if mode == '1':
            # 頻出度順にソートして出力
            results = []
            for token in sorted_tokens:
                # tokenごとにcontextsを抽出
                token_contexts = [
                    ctx for ctx in contexts
                    if ctx["context_words"][ctx["matched_end"]] == token
                ]
                for ctx in token_contexts:
                    ctx_with_token = ctx.copy()
                    ctx_with_token["following_word"] = token
                    results.append(ctx_with_token)
        elif mode == '2':
            for token in sorted_tokens:
                token_results = {
                    "following_word": token,
                    "count": token_counts[token],
                    "contexts": [ctx for ctx in contexts if ctx["context_words"][ctx["matched_end"]] == token]
                }
                results.append(token_results)

    elif mode == '3':
        # query_inputがPOSタグ（NN, NNPなど）の場合、その品詞のみ完全一致で検索
        target_tag = query_input.strip().upper()
        window = 5
        try:
            tagged_words = pos_tag(tokens_original)
        except Exception as e:
            logging.error(f"POS tagging failed: {e}\n{traceback.format_exc()}")
            return jsonify({"error": "POS tagging failed."}), 500

        # タグ一覧をログ出力（デバッグ用）
        tags_in_text = set(tag for _, tag in tagged_words)
        logging.info(f"POS tags in text: {sorted(tags_in_text)}")

        if target_tag not in tags_in_text:
            return jsonify({"results": [], "error": f"POS tag '{query_input}' not found in the text. Available tags: {', '.join(sorted(tags_in_text))}"}), 200

        contexts = []
        freq_counter = Counter()
        for idx, (word, tag) in enumerate(tagged_words):
            # 完全一致のみ
            if tag == target_tag:
                start = max(0, idx - window)
                end = min(len(tagged_words), idx + window + 1)
                context_words = [w for w, t in tagged_words[start:end]]
                matched_start = idx - start
                matched_end = matched_start + 1
                contexts.append({
                    "context_words": context_words,
                    "matched_start": matched_start,
                    "matched_end": matched_end,
                    "center_word": word,
                    "center_index": idx
                })
                freq_counter[word] += 1

        if not contexts:
            return jsonify({"results": [], "error": f"No matches found for POS tag '{query_input}'."}), 200

        sorted_words = [word for word, _ in freq_counter.most_common()]
        results = []
        for word in sorted_words:
            word_contexts = [
                ctx for ctx in contexts if ctx["center_word"] == word
            ]
            results.append({
                "center_word": word,
                "count": freq_counter[word],
                "contexts": word_contexts
            })

        return jsonify({
            "results": results,
            "total_count": sum(freq_counter.values()),
            "language": lang
        })

    return jsonify({"results": results, "language": lang})

@app.route('/api/authorship', methods=['POST'])
def authorship_analysis():
    data = request.json
    url_a = data.get('url_a', '').strip()
    url_b = data.get('url_b', '').strip()

    logging.info(f"Authorship analysis request: URL A={url_a}, URL B={url_b}")

    if not url_a or not url_b:
        return jsonify({"error": "Please provide two Wikipedia URLs."}), 400

    for i, url_val in enumerate([url_a, url_b]):
        label = "A" if i == 0 else "B"
        try:
            parsed = urlparse(url_val)
            if not parsed.scheme in ['http', 'https'] or not parsed.hostname:
                raise ValueError("Invalid URL scheme or hostname.")
            hostname_lower = parsed.hostname.lower()
            if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
                return jsonify({"error": f"URL for Author {label} is not from wikipedia.org."}), 400
        except ValueError as e:
            return jsonify({"error": f"Invalid URL format for Author {label}: {url_val}"}), 400

    try:
        text_a = fetch_wikipedia_text_for_authorship(url_a)
        text_b = fetch_wikipedia_text_for_authorship(url_b)

        if not text_a or not text_b:
            return jsonify({"error": "Failed to fetch content from one or both URLs."}), 500

        sentences_a, labels_a = build_sentence_dataset_for_authorship(text_a, "AuthorA")
        sentences_b, labels_b = build_sentence_dataset_for_authorship(text_b, "AuthorB")

        if not sentences_a or not sentences_b:
            return jsonify({"error": "Could not extract enough valid sentences."}), 400

        all_sentences = sentences_a + sentences_b
        all_labels = labels_a + labels_b

        if len(set(all_labels)) < 2 or len(all_sentences) < 5:
            return jsonify({"error": "Not enough sentences for comparison."}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_mixed_for_authorship,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=10000,
            stop_words=_ALL_SW,
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        distinctive_words = {}
        feature_names = vectorizer.get_feature_names_out()
        for idx, author in enumerate(clf.classes_):
            log_probs = clf.feature_log_prob_[idx]
            top_indices = log_probs.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            distinctive_words[author] = top_terms

        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        report = sk_classification_report(y_test, preds, digits=3, zero_division=0, target_names=sorted(clf.classes_))

        sample_predictions = []
        num_samples = min(len(X_test), 5)
        for sent, true_label, pred_label in zip(X_test[:num_samples], y_test[:num_samples], preds[:num_samples]):
            snippet = (sent[:100] + "…") if len(sent) > 100 else sent
            sample_predictions.append({
                "sentence_snippet": snippet,
                "true_label": true_label,
                "predicted_label": pred_label
            })

        return jsonify({
            "accuracy": f"{acc:.3f}",
            "classification_report": report,
            "distinctive_words": distinctive_words,
            "sample_predictions": sample_predictions,
            "training_samples_count": len(X_train),
            "test_samples_count": len(X_test)
        })

    except ValueError as ve:
        logging.error(f"ValueError: {ve}\n{traceback.format_exc()}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Server error during authorship analysis."}), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=8080, host='0.0.0.0')