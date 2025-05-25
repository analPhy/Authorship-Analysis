# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import shutil # Retained in case it was for a reason, though not used in provided snippets
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
import ssl # Not used in the final version, but kept if there was an original purpose
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys

# --- Authorship Attribution 用の追加インポート ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score

# --- 多言語対応用ライブラリ (Authorship Attribution 用) ---
from langdetect import detect as lang_detect # 名前衝突を避けるためエイリアス
import fugashi
from stopwordsiso import stopwords

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

# NLTKデータ (punkt for tokenization by existing search and authorship)
# This will try to download if not found.
# In production, ensure these are pre-downloaded in your deployment environment.
REQUIRED_NLTK_RESOURCES = ["punkt"]
for res_name in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{res_name}")
        logging.info(f"NLTK resource '{res_name}' found.")
    except (nltk.downloader.DownloadError, LookupError):
        logging.warning(f"NLTK resource '{res_name}' not found. Attempting to download...")
        try:
            nltk.download(res_name)
            logging.info(f"NLTK resource '{res_name}' downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download NLTK resource '{res_name}': {e}")
            # Depending on the resource, this might be a critical failure.

# --- Authorship Attribution 用のグローバル設定 ---
_TAGGER = None
try:
    _TAGGER = fugashi.Tagger()
    logging.info("fugashi Tagger initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize fugashi Tagger: {e}. Japanese tokenization for authorship will not work.")

_EN_SW = set()
_JA_SW = set()
try:
    _EN_SW = stopwords("en")
    _JA_SW = stopwords("ja")
    logging.info("English and Japanese stopwords (for authorship) loaded.")
except Exception as e:
    logging.error(f"Failed to load stopwords (for authorship): {e}")

_ALL_SW = sorted(list(_EN_SW.union(_JA_SW))) # TfidfVectorizerにはリストで渡す

_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")


app = Flask(__name__)

allowed_origins = [
    "http://localhost:3000",
    # "https://your-production-react-app.com", # PRODUCTION: Replace with your frontend domain
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})


# === Text processing for Phrase Search (existing) ===
def get_text_from_url(url):
    logging.info(f"Attempting to fetch URL (for phrase search): {url}")
    try:
        req = urllib.request.Request(
            url,
            data=None,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()):
                 logging.warning(f"URL (phrase search) is not an HTML page: {url} (Content-Type: {content_type})")
                 raise ValueError(f"The provided URL does not appear to be an HTML page. (Content-Type: {content_type})")
            html = response.read()

        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                         'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                         'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched and parsed URL (for phrase search): {url}")
        return text
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching (phrase search) {url}: {e.reason}")
        raise ValueError(f"Could not access the provided URL. Please check the URL. Reason: {e.reason}")
    except TimeoutError:
         logging.error(f"Timeout fetching (phrase search) {url} after 15 seconds.")
         raise ValueError(f"URL fetching timed out. The page took too long to respond.")
    except ValueError as e:
          logging.error(f"Value Error processing (phrase search) {url}: {e}")
          raise ValueError(f"{e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_text_from_url (phrase search) for {url}: {e}\n{traceback.format_exc()}")
        raise ValueError(f"An unexpected error occurred while fetching or processing the URL.")

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Authorship Attribution 用ヘルパー関数 ===
def fetch_wikipedia_text_for_authorship(url: str) -> str:
    logging.info(f"Fetching Wikipedia text for authorship from URL: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Python-Flask-Authorship-App/1.0)'}
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            html = response.read()
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()):
                 logging.warning(f"URL (authorship) is not an HTML page: {url} (Content-Type: {content_type})")
                 raise ValueError(f"The provided URL does not appear to be an HTML page.")
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching (authorship) {url}: {e.reason}")
        raise ValueError(f"Could not access the URL for authorship. Reason: {e.reason}")
    except TimeoutError:
        logging.error(f"Timeout fetching (authorship) {url}")
        raise ValueError("URL fetching for authorship timed out.")
    except Exception as err:
        logging.error(f"Error fetching URL (authorship) {url}: {err}\n{traceback.format_exc()}")
        raise ValueError(f"An unexpected error occurred while fetching URL for authorship.")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "sup", "table", "nav", "footer", "aside", "header", "form", "figure", "figcaption", "link", "meta", "input", "button", "img", "audio", "video", "iframe", "object", "embed", "svg", "canvas", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    logging.info(f"Successfully fetched and parsed Wikipedia text for authorship from URL: {url}")
    return text

def mixed_sentence_tokenize_for_authorship(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def tokenize_mixed_for_authorship(text: str):
    if not _TAGGER:
        logging.warning("Fugashi Tagger not available, defaulting to English tokenization for mixed content (authorship).")
        try: # Fallback, NLTK's word_tokenize should be available
            return word_tokenize(text)
        except Exception as e_tok:
            logging.error(f"NLTK word_tokenize fallback failed: {e_tok}")
            return text.split() # Simplest fallback

    try:
        lang = lang_detect(text)
    except Exception:
        lang = "en"

    if lang == "ja":
        return [tok.surface for tok in _TAGGER(text)]
    else:
        try:
            return word_tokenize(text)
        except Exception as e_tok_en:
            logging.error(f"NLTK word_tokenize for English text failed: {e_tok_en}")
            return text.split() # Simplest fallback

def build_sentence_dataset_for_authorship(text: str, author_label: str, min_len: int = 30):
    sentences = mixed_sentence_tokenize_for_authorship(text)
    filtered = [s for s in sentences if len(s) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels

# === API Endpoints ===
@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    url = data.get('url', '').strip()
    target_input = data.get('phrase', '').strip()

    logging.info(f"Received search request for URL: {url}, Phrase: {target_input}")

    if not url:
        logging.warning("Search request missing URL.")
        return jsonify({"error": "Please provide a Wikipedia URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme received: {url}")
            return jsonify({"error": "Invalid URL scheme. Only http or https are allowed."}), 400
        if not parsed_url.hostname:
             logging.warning(f"Invalid URL hostname received: {url}")
             return jsonify({"error": "Invalid URL hostname."}), 400
        hostname_lower = parsed_url.hostname.lower()
        # Allowing any wikipedia.org subdomain and wikipedia.org itself
        if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
            logging.warning(f"URL not from wikipedia.org: {url}")
            return jsonify({"error": "Only URLs from wikipedia.org are allowed for phrase search."}), 400
    except ValueError:
         logging.warning(f"Invalid URL format received: {url}")
         return jsonify({"error": "Invalid URL format."}), 400
    except Exception as e:
         logging.error(f"An unexpected error during URL parsing: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An unexpected error occurred during URL validation."}), 500

    if not target_input:
        logging.warning("Search request missing phrase.")
        return jsonify({"error": "Please provide a phrase to search."}), 400
    target_words = target_input.split()
    if len(target_words) > 2:
        logging.warning(f"Search phrase too long: {target_input}")
        return jsonify({"error": "Please enter one or two words only."}), 400

    words_from_page = []
    try:
        raw_text = get_text_from_url(url)
        text = clean_text(raw_text)
        if text:
            try:
                words_from_page = word_tokenize(text)
            except Exception as e_tokenize: # Catch if NLTK's punkt is missing despite checks
                logging.error(f"Failed to tokenize text for phrase search (URL: {url}): {e_tokenize}\n{traceback.format_exc()}")
                return jsonify({"error": "Text tokenization failed on the server."}), 500
    except ValueError as e:
        logging.warning(f"Error during URL processing for phrase search {url}: {e}")
        return jsonify({"error": f"{e}"}), 400
    except Exception as e:
         logging.error(f"An unexpected server error during URL processing for phrase search {url}: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An unexpected server error occurred during URL processing."}), 500

    if not words_from_page:
         logging.info(f"No searchable text available for URL (phrase search): {url}")
         return jsonify({
             "results": [],
             "error": f"Could not extract searchable text from the provided URL."
         }), 200

    results = []
    words_lower = [word.lower() for word in words_from_page]
    search_window_size = 5

    if len(target_words) == 1:
        target_word_lower = target_words[0].lower()
        for i, word_lower in enumerate(words_lower):
            if word_lower == target_word_lower:
                before = words_from_page[max(0, i - search_window_size): i]
                after = words_from_page[i + 1: i + 1 + search_window_size]
                context_words = before + [words_from_page[i]] + after
                matched_start_index = len(before)
                matched_end_index = matched_start_index + 1
                results.append({
                    "context_words": context_words,
                    "matched_start": matched_start_index,
                    "matched_end": matched_end_index
                })
    elif len(target_words) == 2:
        target_word1_lower = target_words[0].lower()
        target_word2_lower = target_words[1].lower()
        for i in range(len(words_lower) - 1):
             if words_lower[i] == target_word1_lower and words_lower[i+1] == target_word2_lower:
                before = words_from_page[max(0, i - search_window_size): i]
                after = words_from_page[i + 2: i + 2 + search_window_size]
                context_words = before + [words_from_page[i], words_from_page[i+1]] + after
                matched_start_index = len(before)
                matched_end_index = matched_start_index + 2
                results.append({
                    "context_words": context_words,
                    "matched_start": matched_start_index,
                    "matched_end": matched_end_index
                })

    if not results:
        logging.info(f"Phrase '{target_input}' not found in text from {url}")
        return jsonify({
            "results": [],
            "error": f"The phrase '{target_input}' was not found in the text from the provided URL."
        }), 200
    logging.info(f"Found {len(results)} occurrences for phrase '{target_input}' in text from {url}")
    return jsonify({"results": results})


@app.route('/api/authorship', methods=['POST'])
def authorship_analysis():
    data = request.json
    url_a = data.get('url_a', '').strip()
    url_b = data.get('url_b', '').strip()

    logging.info(f"Received authorship analysis request for URL A: {url_a}, URL B: {url_b}")

    if not url_a or not url_b:
        logging.warning("Authorship request missing one or both URLs.")
        return jsonify({"error": "Please provide two Wikipedia URLs."}), 400

    for i, url_val in enumerate([url_a, url_b]):
        label = "A" if i == 0 else "B"
        try:
            parsed = urlparse(url_val)
            if not parsed.scheme in ['http', 'https'] or not parsed.hostname:
                 raise ValueError("Invalid URL scheme or hostname.")
            # Optional: Restrict to Wikipedia for authorship as well
            hostname_lower = parsed.hostname.lower()
            if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
                logging.warning(f"URL (authorship) not from wikipedia.org: {url_val}")
                return jsonify({"error": f"URL for Author {label} ('{url_val}') is not from wikipedia.org. Only Wikipedia URLs are currently supported."}), 400
        except ValueError as e:
            logging.warning(f"Invalid URL format for authorship (Author {label}): {url_val}. Error: {e}")
            return jsonify({"error": f"Invalid URL format for Author {label}: {url_val}"}), 400

    try:
        logging.info("Downloading Wikipedia pages for authorship...")
        text_a = fetch_wikipedia_text_for_authorship(url_a)
        text_b = fetch_wikipedia_text_for_authorship(url_b)

        if not text_a or not text_b: # Should be caught by exceptions in fetch function
            return jsonify({"error": "Failed to fetch content from one or both Wikipedia pages."}), 500

        logging.info("Splitting into sentences & labelling for authorship...")
        sentences_a, labels_a = build_sentence_dataset_for_authorship(text_a, "AuthorA")
        sentences_b, labels_b = build_sentence_dataset_for_authorship(text_b, "AuthorB")

        if not sentences_a or not sentences_b:
            error_msg = "Could not extract enough valid sentences (min 30 chars) for analysis. "
            if not sentences_a: error_msg += f"Problem with URL A. "
            if not sentences_b: error_msg += f"Problem with URL B. "
            logging.warning(error_msg)
            return jsonify({"error": error_msg.strip()}), 400

        all_sentences = sentences_a + sentences_b
        all_labels = labels_a + labels_b

        if len(set(all_labels)) < 2:
            return jsonify({"error": "Could not gather sentences from both sources to perform comparison."}), 400
        
        # Ensure enough samples for train_test_split with stratify
        # test_size=0.2 means at least 5 samples per class for stratify to typically work well.
        # More generally, n_splits (implicit in test_size) samples per class.
        # The minimum is 2 samples for one class if test_size creates at least 1 sample for test.
        MIN_SAMPLES_PER_CLASS_FOR_STRATIFY = 2 
        if labels_a.count("AuthorA") < MIN_SAMPLES_PER_CLASS_FOR_STRATIFY or \
           labels_b.count("AuthorB") < MIN_SAMPLES_PER_CLASS_FOR_STRATIFY or \
           len(all_sentences) < 5 : # Overall minimum
            logging.warning(f"Not enough samples for stratified split. A: {labels_a.count('AuthorA')}, B: {labels_b.count('AuthorB')}, Total: {len(all_sentences)}")
            return jsonify({"error": "Not enough sentences from each author (or overall) to perform a reliable model training. Please try different URLs with more content."}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        logging.info(f"Authorship Training samples: {len(X_train)} — Test samples: {len(X_test)}")
        if not X_train or not X_test:
            return jsonify({"error": "Failed to create training/testing sets (empty after split). Likely insufficient data."}), 400

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_mixed_for_authorship,
            token_pattern=None, # Crucial when using a custom tokenizer
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
        # Ensure target_names matches clf.classes_ for the report
        report_target_names = sorted(list(clf.classes_)) # Ensures consistent order
        report = sk_classification_report(y_test, preds, digits=3, zero_division=0, target_names=report_target_names)


        sample_predictions = []
        num_samples = min(len(X_test), 5)
        for sent, true_label, pred_label in zip(X_test[:num_samples], y_test[:num_samples], preds[:num_samples]):
            snippet = (sent[:100] + "…") if len(sent) > 100 else sent
            sample_predictions.append({
                "sentence_snippet": snippet,
                "true_label": true_label,
                "predicted_label": pred_label
            })
        
        logging.info(f"Authorship analysis successful. Accuracy: {acc:.3f}")
        return jsonify({
            "accuracy": f"{acc:.3f}",
            "classification_report": report,
            "distinctive_words": distinctive_words,
            "sample_predictions": sample_predictions,
            "training_samples_count": len(X_train),
            "test_samples_count": len(X_test)
        })

    except ValueError as ve:
        logging.error(f"ValueError during authorship analysis: {ve}\n{traceback.format_exc()}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred during authorship analysis: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during authorship analysis."}), 500


if __name__ == "__main__":
    print("Flask development server starting...")
    # Ensure NLTK and Fugashi setup messages appear before Flask starts serving
    if not _TAGGER and "ja" in [_l.lower() for _l in sys.argv]: # Example check
        print("Warning: Fugashi Tagger failed to initialize. Japanese NLP features will be limited.")
    app.run(debug=True, port=5000, host='0.0.0.0')