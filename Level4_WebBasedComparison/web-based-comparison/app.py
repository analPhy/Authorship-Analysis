# --- Imports ---
# EN: Import necessary libraries for web server, text processing, ML, etc.
# JP: Webサーバー、テキスト処理、機械学習などに必要なライブラリをインポート
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('maxent_ne_chunker_tab')
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
from collections import Counter
from langdetect import detect as lang_detect
import fugashi
from stopwordsiso import stopwords

# --- Authorship Attribution Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score

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
        logging.warning(f"NLTK resource '{download_key}' not found. Attempting to download...")
        try:
            download_successful = nltk.download(download_key, quiet=False)
            if download_successful:
                logging.info(f"NLTK download for '{download_key}' executed successfully.")
                nltk.data.find(path_to_find)
                logging.info(f"NLTK resource '{download_key}' now found.")
            else:
                logging.error(f"NLTK download for '{download_key}' failed. Verifying anyway...")
                try:
                    nltk.data.find(path_to_find)
                    logging.info(f"NLTK resource '{download_key}' found despite download command returning False.")
                except LookupError:
                    logging.error(f"CRITICAL: NLTK resource '{download_key}' still not found. Manual intervention required.")
        except Exception as e_download:
            logging.error(f"Error during NLTK resource '{download_key}' download: {e_download}")
            print(f"Please run: python -m nltk.downloader {download_key}")

# --- Global Settings for Authorship Attribution ---
_TAGGER = None
try:
    _TAGGER = fugashi.Tagger()
    logging.info("fugashi Tagger initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize fugashi Tagger: {e}")
    print("Warning: Japanese NLP features will be limited.")

_EN_SW = set()
_JA_SW = set()
try:
    _EN_SW = stopwords("en")
    _JA_SW = stopwords("ja")
    logging.info("English and Japanese stopwords loaded.")
except Exception as e:
    logging.error(f"Failed to load stopwords: {e}")

_ALL_SW = sorted(list(_EN_SW.union(_JA_SW)))

# EN: Regex for sentence splitting (Japanese and English)
# JP: 文分割用の正規表現（日本語・英語対応）
_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# --- Flask App Setup ---
app = Flask(__name__)
allowed_origins = [
    "http://localhost:3000",
    # "https://your-production-react-app.com", # PRODUCTION: Replace with your frontend domain
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# === Text processing for KWIC Search ===
# EN: Fetch and clean text from a URL, aligned with level2.py's get_clean_web_text
# JP: level2.pyのget_clean_web_textと一致するテキスト取得・クリーンアップ
def get_text_from_url_for_kwic(url):
    logging.info(f"Fetching URL for KWIC search: {url}")
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme: {url}")
            return None
        if not parsed_url.hostname:
            logging.warning(f"Invalid URL hostname: {url}")
            return None
        hostname_lower = parsed_url.hostname.lower()
        if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
            logging.warning(f"URL not from wikipedia.org: {url}")
            return None

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
                logging.warning(f"URL is not an HTML page: {url} (Content-Type: {content_type})")
                return None
            html = response.read()

        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                         'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                         'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        logging.info(f"Successfully fetched and cleaned text from URL: {url}")
        return text
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching {url}: {e.reason}")
        return None
    except TimeoutError:
        logging.error(f"Timeout fetching {url} after 15 seconds.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {e}\n{traceback.format_exc()}")
        return None

# EN: Clean text for KWIC search, aligned with level2.py
# JP: level2.pyと一致するKWIC検索用のテキストクリーンアップ
def clean_text_for_kwic(text):
    if not text:
        return ""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# EN: Tokenize text based on language, aligned with level2.py's tokenize_mixed
# JP: level2.pyのtokenize_mixedと一致する言語ベースのトークン化
def tokenize_mixed_for_authorship(text, lang="en"):
    PUNCT_SKIP = {".", ",", "(", ")", "'"}
    try:
        if lang == "ja" and _TAGGER:
            tokens = [tok.surface for tok in _TAGGER(text) if tok.feature.pos1 not in ['助詞', '助動詞', '記号']]
        else:
            tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in PUNCT_SKIP]
        return tokens
    except Exception as e:
        logging.error(f"Tokenization error: {e}")
        return text.split()  # Fallback to simple split

# === API Endpoints ===

@app.route('/api/search', methods=['POST'])
def kwic_search():
    # EN: KWIC search API endpoint, aligned with level2.py's find_following_words
    # JP: level2.pyのfind_following_wordsと一致するKWIC検索APIエンドポイント
    data = request.json
    url = data.get('url', '').strip()
    target = data.get('target', '').strip()
    mode = data.get('mode', '').strip()
    context_window = 5  # Default context window size, same as level2.py

    logging.info(f"Received KWIC search request. URL: {url}, Target: '{target}', Mode: {mode}")

    # EN: Input validation
    # JP: 入力値のバリデーション
    if not url:
        logging.warning("KWIC search request missing URL.")
        return jsonify({"error": "Please provide a Wikipedia URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme: {url}")
            return jsonify({"error": "Invalid URL scheme. Only http or https are allowed."}), 400
        if not parsed_url.hostname:
            logging.warning(f"Invalid URL hostname: {url}")
            return jsonify({"error": "Invalid URL hostname."}), 400
        hostname_lower = parsed_url.hostname.lower()
        if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
            logging.warning(f"URL not from wikipedia.org: {url}")
            return jsonify({"error": "URL is not from wikipedia.org. Only Wikipedia URLs are supported."}), 400
    except ValueError:
        logging.warning(f"Invalid URL format: {url}")
        return jsonify({"error": "Invalid URL format."}), 400
    except Exception as e:
        logging.error(f"Unexpected error during URL parsing: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected error occurred during URL validation."}), 500

    if not target:
        logging.warning("KWIC search request missing target phrase.")
        return jsonify({"error": "Please provide a target phrase."}), 400
    target_words = target.split()
    if not 1 <= len(target_words) <= 2:
        logging.warning(f"Invalid target phrase length: {target} (Length: {len(target_words)})")
        return jsonify({"error": "Please enter 1 or 2 words for the target phrase."}), 400

    if mode not in ['1', '2', '3']:
        logging.warning(f"Invalid mode: {mode}")
        return jsonify({"error": "Invalid mode. Please choose 1 (Sequential), 2 (Token), or 3 (POS)."}), 400

    # EN: Fetch and process text
    # JP: テキストを取得・処理
    text = get_text_from_url_for_kwic(url)
    if not text:
        logging.warning(f"No text retrieved from URL: {url}")
        return jsonify({"error": "Could not extract text from the provided URL."}), 400

    text_cleaned = clean_text_for_kwic(text)
    if not text_cleaned:
        logging.info(f"No searchable text available for URL: {url}")
        return jsonify({
            "results": [],
            "error": "Could not extract searchable text from the provided URL."
        }), 200

    # EN: Detect language
    # JP: 言語を検出
    try:
        lang = lang_detect(text_cleaned) if text_cleaned else "en"
    except Exception:
        lang = "en"
    logging.info(f"Detected language: {lang}")

    # EN: Tokenize text
    # JP: テキストをトークン化
    try:
        tokens_original = tokenize_mixed_for_authorship(text_cleaned, lang)
        tokens_lower = [t.lower() for t in tokens_original]
    except Exception as e:
        logging.error(f"Tokenization failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Text tokenization failed on the server."}), 500

    if not tokens_original:
        logging.info(f"No tokens extracted from URL: {url}")
        return jsonify({
            "results": [],
            "error": "Could not extract tokens from the provided URL."
        }), 200

    # EN: Find following words, aligned with level2.py's find_following_words
    # JP: level2.pyのfind_following_wordsと一致する後続単語の検索
    target_str = ' '.join(word.lower() for word in target_words)
    target_token_list = target_str.split()
    num_target_tokens = len(target_token_list)

    following_words = []
    contexts = []

    for i in range(len(tokens_lower) - num_target_tokens + 1):
        if tokens_lower[i:i + num_target_tokens] == target_token_list:
            next_idx = i + num_target_tokens
            if next_idx < len(tokens_lower):
                next_token = tokens_original[next_idx]  # Original case for display
                before = tokens_original[max(0, i - context_window):i]
                after = tokens_original[next_idx + 1:next_idx + 1 + context_window]
                before_str = ' '.join(before)
                matched_str = ' '.join(tokens_original[i:i + num_target_tokens])
                after_str = ' '.join(after)
                following_words.append(next_token)
                contexts.append({
                    "before": before_str,
                    "target": matched_str,
                    "following": next_token,
                    "after": after_str
                })

    if not following_words:
        logging.info(f"No matches found for target '{target_str}' in text from {url}")
        return jsonify({
            "results": [],
            "error": f"No matches found for the target phrase '{target_str}'."
        }), 200

    # EN: Prepare results based on mode
    # JP: モードに基づいて結果を準備
    results = []
    token_counts = Counter(following_words)
    sorted_tokens = [token for token, _ in token_counts.most_common()]  # 頻度順にソート

    if mode == '1':
        # Sequential: Return contexts in order of appearance
        results = contexts
    elif mode == '2':
        # Token: Group by following word, sorted by frequency
        for token in sorted_tokens:
            token_results = {
                "following_word": token,
                "count": token_counts[token],
                "contexts": [ctx for ctx in contexts if ctx["following"] == token]
            }
            results.append(token_results)
    elif mode == '3':
        # POS: Group by POS tags, sorted by token frequency
        try:
            if lang == "ja" and _TAGGER:
                tagged_tokens = [(tok.surface, tok.feature.pos1) for tok in _TAGGER(' '.join(following_words))]
            else:
                tagged_tokens = pos_tag(following_words)
            token_to_pos = dict(tagged_tokens)
            sorted_tagged = sorted(tagged_tokens, key=lambda x: token_counts[x[0]], reverse=True)
            seen = set()
            for word, tag in sorted_tagged:
                if word not in seen:
                    seen.add(word)
                    pos_results = {
                        "pos_tag": tag,
                        "following_word": word,
                        "count": token_counts[word],
                        "contexts": [ctx for ctx in contexts if ctx["following"] == word]
                    }
                    results.append(pos_results)
        except Exception as e:
            logging.error(f"POS tagging failed: {e}\n{traceback.format_exc()}")
            return jsonify({"error": "Part-of-speech tagging failed on the server."}), 500

    logging.info(f"Found {len(following_words)} occurrences for target '{target_str}' (Mode: {mode}) in text from {url}")
    return jsonify({
        "results": results,
        "language": lang,
        "total_matches": len(following_words)
    })

# === Authorship Attribution Helpers ===
# ... (既存のコードはそのまま保持。fetch_wikipedia_text_for_authorship, mixed_sentence_tokenize_for_authorship, build_sentence_dataset_for_authorship は変更なし)

@app.route('/api/authorship', methods=['POST'])
def authorship_analysis():
    # ... (既存のコードはそのまま。変更なし)
    # 必要に応じて、以下は元のコードをそのまま使用
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

        if not text_a or not text_b:
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
        
        MIN_SAMPLES_PER_CLASS_FOR_STRATIFY = 2 
        if labels_a.count("AuthorA") < MIN_SAMPLES_PER_CLASS_FOR_STRATIFY or \
           labels_b.count("AuthorB") < MIN_SAMPLES_PER_CLASS_FOR_STRATIFY or \
           len(all_sentences) < 5 :
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
        report_target_names = sorted(list(clf.classes_))
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
        logging.error(f"Unexpected error during authorship analysis: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during authorship analysis."}), 500

if __name__ == "__main__":
    print("Flask development server starting...")
    print("Verifying NLTK resources...")
    all_nltk_res_ok = True
    for res_key in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(REQUIRED_NLTK_RESOURCES[res_key])
        except LookupError:
            print(f"Warning: NLTK resource '{res_key}' was not found or could not be downloaded.")
            print(f"KWIC search functionality involving this resource (e.g., '{res_key}') may be impaired.")
            all_nltk_res_ok = False
    if all_nltk_res_ok:
        print("All required NLTK resources appear to be available.")

    if not _TAGGER:
        print("Warning: Fugashi Tagger failed to initialize. Japanese NLP features will be limited.")
    
    print("Starting Flask app server...")
    app.run(debug=True, port=8080, host='0.0.0.0')