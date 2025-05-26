# app.py


# --- Imports ---
# EN: Import necessary libraries for web server, text processing, ML, etc.
# JP: Webサーバー、テキスト処理、機械学習などに必要なライブラリをインポート
from flask import Flask, jsonify, request
from flask_cors import CORS
# import shutil
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
# import ssl
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys

# --- Authorship Attribution Imports ---
# EN: Import scikit-learn modules for authorship analysis
# JP: 著者識別分析用のscikit-learnモジュールをインポート
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score

# --- Multilingual Support Libraries ---
# EN: Import libraries for language detection, Japanese tokenization, and stopwords
# JP: 言語判定、日本語形態素解析、ストップワード用ライブラリをインポート
from langdetect import detect as lang_detect
import fugashi
from stopwordsiso import stopwords

# --- Logging Setup ---
# EN: Configure logging for debugging and monitoring
# JP: デバッグや監視用のロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

# --- NLTK Resource Check ---
# EN: Check and download required NLTK resources if missing
# JP: 必要なNLTKリソースがなければダウンロード
REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",  # 標準の文分割モデル
    "punkt_tab": "tokenizers/punkt_tab", # エラーメッセージに基づき追加
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words"
}

logging.info("Checking NLTK resources...")
for download_key, path_to_find in REQUIRED_NLTK_RESOURCES.items():
    try:
        nltk.data.find(path_to_find)
        logging.info(f"NLTK resource '{download_key}' (for path '{path_to_find}') found.")
    except LookupError:
        logging.warning(f"NLTK resource '{download_key}' (for path '{path_to_find}') not found. Attempting to download '{download_key}'...")
        try:
            # EN: Attempt to download the resource
            # JP: リソースのダウンロードを試みる
            download_successful = nltk.download(download_key, quiet=False)
            
            if download_successful:
                logging.info(f"NLTK download command for '{download_key}' executed, appearing successful.")
                nltk.data.find(path_to_find)
                logging.info(f"NLTK resource '{download_key}' now found after download.")
            else:
                logging.error(f"NLTK download for '{download_key}' returned False. Attempting to verify if it's available anyway...")
                try:
                    nltk.data.find(path_to_find)
                    logging.info(f"NLTK resource '{download_key}' found despite download command returning False (perhaps already existed or downloaded by other means).")
                except LookupError:
                    logging.error(f"CRITICAL: NLTK resource '{download_key}' still not found after download command returned False. Manual intervention likely required.")
        
        except Exception as e_download:
            logging.error(f"Error during NLTK resource '{download_key}' download or verification: {e_download}")
            logging.error(f"Please check network connectivity and NLTK setup. You may need to run: python -m nltk.downloader {download_key}")

# --- Global Settings for Authorship Attribution ---
# EN: Initialize Japanese tokenizer and stopwords
# JP: 日本語トークナイザーとストップワードの初期化
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

_ALL_SW = sorted(list(_EN_SW.union(_JA_SW)))

# EN: Regex for sentence splitting (Japanese and English)
# JP: 文分割用の正規表現（日本語・英語対応）
_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# --- Flask App Setup ---
# EN: Create Flask app and configure CORS
# JP: Flaskアプリの作成とCORS設定
app = Flask(__name__)

# app.py のCORS設定部分を以下のように変更

# GitHub PagesのURLを特定します。
## --- CORS 設定 ---
# EN: Define allowed origins for CORS
# JP: CORSを許可するオリジンを定義
FRONTEND_GITHUB_PAGES_ORIGIN = "https://analphy.github.io"  # あなたのGitHub Pagesのオリジン
FRONTEND_DEV_ORIGIN_3000 = "http://localhost:3000"         # React開発サーバーの一般的なポート
FRONTEND_DEV_ORIGIN_8080 = "http://localhost:8080"         # あなたがフロントエンド開発で使っていたポート

allowed_origins_list = [
    FRONTEND_GITHUB_PAGES_ORIGIN,
    FRONTEND_DEV_ORIGIN_3000,
    FRONTEND_DEV_ORIGIN_8080,
]

CORS(app, resources={r"/api/*": {"origins": allowed_origins_list}})
# --- ここまでCORS設定 ---

# === Text processing for KWIC Search ===
# EN: Fetch and clean text from a Wikipedia URL for KWIC search
# JP: KWIC検索用にWikipediaのURLからテキストを取得・整形
def get_text_from_url_for_kwic(url):
    logging.info(f"Attempting to fetch URL (for KWIC search): {url}")
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
                 logging.warning(f"URL (KWIC search) is not an HTML page: {url} (Content-Type: {content_type})")
                 raise ValueError(f"The provided URL does not appear to be an HTML page. (Content-Type: {content_type})")
            html = response.read()

        soup = BeautifulSoup(html, 'html.parser')
        # EN: Remove unnecessary tags
        # JP: 不要なタグを除去
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                         'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                         'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched and parsed URL (for KWIC search): {url}")
        return text
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching (KWIC search) {url}: {e.reason}")
        raise ValueError(f"Could not access the provided URL. Please check the URL. Reason: {e.reason}")
    except TimeoutError:
         logging.error(f"Timeout fetching (KWIC search) {url} after 15 seconds.")
         raise ValueError(f"URL fetching timed out. The page took too long to respond.")
    except ValueError as e:
          logging.error(f"Value Error processing (KWIC search) {url}: {e}")
          raise
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_text_from_url_for_kwic for {url}: {e}\n{traceback.format_exc()}")
        raise ValueError(f"An unexpected error occurred while fetching or processing the URL.")

def clean_text_for_kwic(text):
    # EN: Remove citation numbers and extra spaces
    # JP: 参照番号や余分な空白を除去
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Authorship Attribution Helpers ===
# EN: Fetch and clean Wikipedia text for authorship analysis
# JP: 著者識別分析用にWikipediaテキストを取得・整形
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
    # EN: Remove unnecessary tags
    # JP: 不要なタグを除去
    for tag in soup(["script", "style", "sup", "table", "nav", "footer", "aside", "header", "form", "figure", "figcaption", "link", "meta", "input", "button", "img", "audio", "video", "iframe", "object", "embed", "svg", "canvas", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    logging.info(f"Successfully fetched and parsed Wikipedia text for authorship from URL: {url}")
    return text

def mixed_sentence_tokenize_for_authorship(text: str):
    # EN: Split text into sentences (supports Japanese and English)
    # JP: テキストを文ごとに分割（日本語・英語対応）
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

# app.py の tokenize_mixed_for_authorship 関数を以下のように置き換える

def tokenize_mixed_for_authorship(text: str):
    # EN: Always use English word_tokenize for simplicity and performance on Render free tier.
    # JP: Render無料枠でのシンプルさとパフォーマンスのため、常に英語のword_tokenizeを使用する。
    #     言語判定とFugashiの使用を一時的にコメントアウト。
    # if not _TAGGER:
    #     logging.warning("Fugashi Tagger not available, defaulting to English tokenization for mixed content (authorship).")
    #     try:
    #         return word_tokenize(text)
    #     except Exception as e_tok:
    #         logging.error(f"NLTK word_tokenize fallback failed: {e_tok}")
    #         return text.split()

    # try:
    #     # This lang_detect call seems to be causing timeouts
    #     lang = lang_detect(text)
    # except Exception:
    #     lang = "en" # Default to English if lang_detect fails

    # if lang == "ja" and _TAGGER: # Ensure _TAGGER is available
    #     return [tok.surface for tok in _TAGGER(text)]
    # else:
    #     # Default to English tokenization
    try:
        # Consider lowercasing here if not done elsewhere consistently for TF-IDF
        return word_tokenize(text) # text.lower() を適用するかはTF-IDFの preprocess との兼ね合い
    except Exception as e_tok_en:
        logging.error(f"NLTK word_tokenize for English text failed: {e_tok_en}")
        return text.split() # text.lower().split()
    
def build_sentence_dataset_for_authorship(text: str, author_label: str, min_len: int = 30):
    # EN: Build a dataset of sentences and labels for authorship analysis
    # JP: 著者識別分析用の文とラベルのデータセットを作成
    sentences = mixed_sentence_tokenize_for_authorship(text)
    filtered = [s for s in sentences if len(s) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels

# === API Endpoints ===

@app.route('/api/search', methods=['POST'])
def kwic_search():
    # EN: KWIC search API endpoint
    # JP: KWIC検索APIエンドポイント
    data = request.json
    url = data.get('url', '').strip()
    query_input = data.get('query', '').strip()
    search_type = data.get('type', 'token').strip().lower()

    logging.info(f"Received KWIC search request. URL: {url}, Query: '{query_input}', Type: {search_type}")

    # EN: Input validation
    # JP: 入力値のバリデーション
    if not url:
        logging.warning("KWIC search request missing URL.")
        return jsonify({"error": "Please provide a Wikipedia URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme for KWIC search: {url}")
            return jsonify({"error": "Invalid URL scheme. Only http or https are allowed."}), 400
        if not parsed_url.hostname:
             logging.warning(f"Invalid URL hostname for KWIC search: {url}")
             return jsonify({"error": "Invalid URL hostname."}), 400
    except ValueError:
         logging.warning(f"Invalid URL format for KWIC search: {url}")
         return jsonify({"error": "Invalid URL format."}), 400
    except Exception as e:
         logging.error(f"An unexpected error during URL parsing for KWIC search: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An unexpected error occurred during URL validation."}), 500

    if not query_input:
        logging.warning("KWIC search request missing query.")
        return jsonify({"error": "Please provide a search query."}), 400

    # EN: Prepare query for each search type
    # JP: 検索タイプごとにクエリを準備
    target_tokens = [] # Initialize for token search
    if search_type == 'token':
        target_tokens = query_input.split()
        if not 1 <= len(target_tokens) <= 5:
            logging.warning(f"Invalid token query length: {query_input} (Length: {len(target_tokens)})")
            return jsonify({"error": "For token search, please enter one to five words."}), 400
    elif search_type in ['pos', 'entity']:
        if " " in query_input or not query_input:
            logging.warning(f"Invalid {search_type} query: '{query_input}'. Should be a single tag/type.")
            return jsonify({"error": f"For {search_type} search, please enter a single valid tag/type (no spaces)."}), 400
    else:
        logging.warning(f"Invalid search_type: {search_type}")
        return jsonify({"error": f"Invalid search type: {search_type}. Supported types are 'token', 'pos', 'entity'."}), 400

    words_from_page_original_case = []
    try:
        raw_text = get_text_from_url_for_kwic(url)
        text_cleaned = clean_text_for_kwic(raw_text)
        if text_cleaned:
            try:
                words_from_page_original_case = word_tokenize(text_cleaned)
            except Exception as e_tokenize:
                logging.error(f"Failed to tokenize text for KWIC search (URL: {url}): {e_tokenize}\n{traceback.format_exc()}")
                return jsonify({"error": "Text tokenization failed on the server."}), 500
    except ValueError as e:
        logging.warning(f"Error during URL processing for KWIC search {url}: {e}")
        return jsonify({"error": f"{e}"}), 400
    except Exception as e:
         logging.error(f"An unexpected server error during URL processing for KWIC search {url}: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An unexpected server error occurred during URL processing."}), 500

    if not words_from_page_original_case:
         logging.info(f"No searchable text available for URL (KWIC search): {url}")
         return jsonify({
             "results": [],
             "error": f"Could not extract searchable text from the provided URL."
         }), 200

    results = []
    backend_context_window_size = 10

    # EN: KWIC search logic for each type
    # JP: 各検索タイプごとのKWIC検索ロジック
    if search_type == 'token':
        words_from_page_lower = [w.lower() for w in words_from_page_original_case]
        target_token_list_lower = [word.lower() for word in target_tokens]
        num_target_tokens = len(target_token_list_lower)

        for i in range(len(words_from_page_lower) - num_target_tokens + 1):
            if words_from_page_lower[i : i + num_target_tokens] == target_token_list_lower:
                before = words_from_page_original_case[max(0, i - backend_context_window_size): i]
                matched_segment = words_from_page_original_case[i : i + num_target_tokens]
                after = words_from_page_original_case[i + num_target_tokens : i + num_target_tokens + backend_context_window_size]
                
                context_words_list = before + matched_segment + after
                result_matched_start = len(before)
                result_matched_end = result_matched_start + num_target_tokens
                
                results.append({
                    "context_words": context_words_list,
                    "matched_start": result_matched_start,
                    "matched_end": result_matched_end
                })

    elif search_type == 'pos':
        try:
            tagged_words = nltk.pos_tag(words_from_page_original_case)
        except Exception as e_pos_tag:
            logging.error(f"NLTK pos_tag failed: {e_pos_tag}\n{traceback.format_exc()}")
            return jsonify({"error": "Part-of-speech tagging failed on the server."}), 500
            
        target_pos_tag_query = query_input.upper()

        for i, (word, tag) in enumerate(tagged_words):
            if tag == target_pos_tag_query:
                before = words_from_page_original_case[max(0, i - backend_context_window_size): i]
                matched_word = [words_from_page_original_case[i]]
                after = words_from_page_original_case[i + 1 : i + 1 + backend_context_window_size]
                
                context_words_list = before + matched_word + after
                result_matched_start = len(before)
                result_matched_end = result_matched_start + 1
                
                results.append({
                    "context_words": context_words_list,
                    "matched_start": result_matched_start,
                    "matched_end": result_matched_end
                })
                
    elif search_type == 'entity':
        try:
            tagged_words_for_ner = nltk.pos_tag(words_from_page_original_case)
            chunked_entities_tree = nltk.ne_chunk(tagged_words_for_ner)
            iob_tags = nltk.chunk.util.tree2conlltags(chunked_entities_tree)
        except Exception as e_ner:
            logging.error(f"NLTK entity recognition failed: {e_ner}\n{traceback.format_exc()}")
            return jsonify({"error": "Entity recognition processing failed on the server."}), 500

        target_entity_type_query = query_input.upper()
        
        idx = 0
        while idx < len(iob_tags):
            word_original, _, iob_label = iob_tags[idx]
            
            if iob_label.startswith('B-') and iob_label[2:] == target_entity_type_query:
                current_entity_original_words = [word_original]
                entity_start_index_in_page = idx
                
                next_idx = idx + 1
                while next_idx < len(iob_tags):
                    next_word_original, _, next_iob_label = iob_tags[next_idx]
                    if next_iob_label.startswith('I-') and next_iob_label[2:] == target_entity_type_query:
                        current_entity_original_words.append(next_word_original)
                        next_idx += 1
                    else:
                        break
                
                num_entity_tokens = len(current_entity_original_words)
                before = words_from_page_original_case[max(0, entity_start_index_in_page - backend_context_window_size): entity_start_index_in_page]
                after = words_from_page_original_case[next_idx : next_idx + backend_context_window_size]
                
                context_words_list = before + current_entity_original_words + after
                result_matched_start = len(before)
                result_matched_end = result_matched_start + num_entity_tokens
                
                results.append({
                    "context_words": context_words_list,
                    "matched_start": result_matched_start,
                    "matched_end": result_matched_end
                })
                idx = next_idx
            else:
                idx += 1

    if not results:
        logging.info(f"Query '{query_input}' (Type: {search_type}) not found in text from {url}")
        return jsonify({
            "results": [],
            "error": f"The query '{query_input}' (Type: {search_type}) was not found in the text from the provided URL."
        }), 200
        
    logging.info(f"Found {len(results)} occurrences for query '{query_input}' (Type: {search_type}) in text from {url}")
    return jsonify({"results": results})

@app.route('/api/authorship', methods=['POST'])
def authorship_analysis(): # No changes to this function
    # EN: Authorship analysis API endpoint
    # JP: 著者識別分析APIエンドポイント
    data = request.json
    url_a = data.get('url_a', '').strip()
    url_b = data.get('url_b', '').strip()

    logging.info(f"Received authorship analysis request for URL A: {url_a}, URL B: {url_b}")

    # EN: Input validation
    # JP: 入力値のバリデーション
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

        # EN: Vectorize sentences using TF-IDF (with n-grams and stopwords)
        # JP: TF-IDF（N-gram・ストップワード対応）で文をベクトル化
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_mixed_for_authorship,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=10000,
            stop_words=_ALL_SW,
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # EN: Train Naive Bayes classifier
        # JP: ナイーブベイズ分類器で学習
        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        # EN: Extract top distinctive words for each author
        # JP: 各著者の特徴的な単語を抽出
        distinctive_words = {}
        feature_names = vectorizer.get_feature_names_out()
        for idx, author in enumerate(clf.classes_):
            log_probs = clf.feature_log_prob_[idx]
            top_indices = log_probs.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            distinctive_words[author] = top_terms

        # EN: Predict and evaluate
        # JP: 予測と評価
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        report_target_names = sorted(list(clf.classes_))
        report = sk_classification_report(y_test, preds, digits=3, zero_division=0, target_names=report_target_names)

        # EN: Prepare sample predictions for frontend
        # JP: フロントエンド用のサンプル予測を準備
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
    # EN: Startup checks and Flask app launch
    # JP: 起動時のチェックとFlaskアプリの起動
    print("Flask development server starting...")
    print("Verifying NLTK resources (this may take a moment if downloads are needed)...")
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
        print("Warning: Fugashi Tagger failed to initialize. Japanese NLP features for authorship attribution will be limited.")
    
    print("Starting Flask app server...")
    app.run(debug=True, port=8080, host='0.0.0.0')