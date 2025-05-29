# --- Imports ---
# EN: Import necessary libraries for web server, text processing, ML, etc.
# JP: Webサーバー、テキスト処理、機械学習などに必要なライブラリをインポート
from collections import Counter
from flask import Flask, jsonify, request
from flask_cors import CORS
# import shutil
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
nltk.download('maxent_ne_chunker_tab')
# import ssl
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
import string
import subprocess


# app.py の冒頭 (import の後など)
import nltk
import sys
import os

# 環境変数 NLTK_DATA からパスを取得。なければデフォルトで固定パスを使用。
expected_nltk_data_dir = os.environ.get('NLTK_DATA', '/opt/render/project/src/nltk_data_on_render')
print(f"--- [APP STARTUP] Expected NLTK_DATA directory: {expected_nltk_data_dir} (from ENV or default) ---")

# nltk.data.path にこの期待されるパスが含まれているか確認し、なければ先頭に追加
if not os.path.isdir(expected_nltk_data_dir):
     print(f"--- [APP STARTUP] CRITICAL ERROR: NLTK data directory '{expected_nltk_data_dir}' does NOT exist. ---")
     sys.exit(1)

if expected_nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, expected_nltk_data_dir)

print(f"--- [APP STARTUP] NLTK data path being used: {nltk.data.path} ---")

# Build Command の download_nltk.py と完全に同じリストにする
required_nltk_resources_app = {
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'punkt': 'tokenizers/punkt',
    'words': 'corpora/words',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
    'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab'
    # 'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger' # 必要なら
}
all_resources_available_app = True

print("--- [APP STARTUP] Checking NLTK resources availability... ---")
for res_id, res_check_path in required_nltk_resources_app.items():
    try:
        nltk.data.find(res_check_path) # ここはres_check_path (辞書のキーではなく値)
        print(f"--- [APP STARTUP]   [FOUND] NLTK resource for {res_id} (Path: {res_check_path}) ---")
    except LookupError:
        print(f"--- [APP STARTUP]   [CRITICAL NOT FOUND] NLTK resource for {res_id} (Path: {res_check_path}) ---")
        all_resources_available_app = False

if not all_resources_available_app:
    print("--- [APP STARTUP] CRITICAL ERROR: Not all required NLTK resources were found during app startup. ---")
    print("--- [APP STARTUP] This indicates an issue with the build process or NLTK_DATA path. ---")
    print("--- [APP STARTUP] Exiting application. ---")
    sys.exit(1) # リソースが不足していればアプリケーションを起動させない
else:
    print("--- [APP STARTUP] All required NLTK resources are available for the application. ---")

# spaCyモデルのロード確認も同様に
try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
    print("--- [APP STARTUP] spaCy model 'en_core_web_sm' loaded successfully. ---")
except OSError as e:
    print(f"--- [APP STARTUP] CRITICAL ERROR: spaCy model 'en_core_web_sm' not found or failed to load: {e} ---")
    sys.exit(1)

# --- ここからFlaskアプリケーションの定義など ---
print("--- [APP STARTUP] Initializing Flask application... ---")



model_name = "en_core_web_sm" # 使用したいモデル名

try:
    spacy.load(model_name)
    print(f"Model '{model_name}' already installed and loaded.")
except OSError:
    print(f"Model '{model_name}' not found. Downloading and installing...")
    try:
        subprocess.check_call(['python', '-m', 'spacy', 'download', model_name])
        # subprocess.check_call([sys.executable, '-m', 'spacy', 'download', model_name]) # より確実な場合
        spacy.load(model_name) # ダウンロード後にロードを試みる
        print(f"Model '{model_name}' downloaded and loaded successfully.")
    except Exception as e:
        print(f"Error downloading or loading model '{model_name}': {e}")

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
PUNCTUATION_SET = set(string.punctuation + '。、「」』『【】・（）　')
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
                # First remove any generic elements we don't want
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                        'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                        'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()

        # EN: Find and clean the main content
        # JP: メインコンテンツを探して整形
        content = soup.find(id='mw-content-text')
        if content:
            # Remove navigation, references, metadata sections and other Wikipedia elements
            wiki_elements = ['toc', 'reference', 'reflist', 'navbox', 'metadata', 'catlinks', 
                           'mw-editsection', 'mw-references', 'mw-navigation', 'mw-footer',
                           'sistersitebox', 'noprint', 'mw-jump-to-nav', 'mw-indicator',
                           'mw-wiki-logo', 'mw-page-tools', 'printfooter', 'mw-revision']

            for element in content.find_all(['div', 'section', 'span', 'nav', 'footer']):
                if any(cls in (element.get('class', []) or []) for cls in wiki_elements):
                    element.decompose()

            text = content.get_text(separator=' ')
        else:
            # Fallback to full page if main content section not found
            text = soup.get_text(separator=' ')
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
    # EN: Remove Wikipedia-specific content and clean text
    # JP: Wikipedia特有のコンテンツを除去してテキストを整形

    # Remove navigation elements and metadata
    text = re.sub(r'Jump to.*?content', '', text)
    text = re.sub(r'From Wikipedia.*?encyclopedia', '', text)
    text = re.sub(r'Categories\s*:.*', '', text)
    text = re.sub(r'Hidden categories\s*:.*', '', text)

    # Remove common Wikipedia elements
    text = re.sub(r'\[\d+\]', '', text)  # citations
    text = re.sub(r'Edit.*?section', '', text)  # edit section links
    text = re.sub(r'oldid=\d+', '', text)  # oldid references
    text = re.sub(r'\s*\(\s*disambiguation\s*\)\s*', '', text)  # disambiguation notes
    text = re.sub(r'CS1.*?sources.*$', '', text, flags=re.MULTILINE)  # CS1 source notes
    text = re.sub(r'Articles.*?containing.*$', '', text, flags=re.MULTILINE)  # Article metadata
    text = re.sub(r'Use mdy dates.*$', '', text, flags=re.MULTILINE)  # Date format notes

    # Remove any remaining Category tags
    text = re.sub(r'Category:.*?(?=\n|$)', '', text)

    # Clean up extra whitespace
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
    for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                     'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                     'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
        tag.decompose()

    # EN: Find and clean the main content
    # JP: メインコンテンツを探して整形
    content = soup.find(id='mw-content-text')
    if content:
        # Remove navigation, references, and metadata sections
        for section in content.find_all(['div', 'section'], class_=['toc', 'reference', 'reflist', 'navbox', 'metadata', 'catlinks']):
            section.decompose()

        # Remove specific Wikipedia elements
        for element in content.find_all(['span', 'div'], class_=['mw-editsection', 'reference', 'reflist']):
            element.decompose()

        text = content.get_text(separator=' ')
    else:
        # Fallback to full page if main content section not found
        text = soup.get_text(separator=' ')

    # Clean up the text
    text = re.sub(r'\[\d+\]', '', text)  # citations
    text = re.sub(r'Edit.*?section', '', text)  # edit section links
    text = re.sub(r'Jump to.*?content', '', text)  # navigation elements
    text = re.sub(r'From Wikipedia.*?encyclopedia', '', text)  # wiki header
    text = re.sub(r'Categories\s*:.*', '', text)  # categories
    text = re.sub(r'\s+', ' ', text).strip()
    logging.info(f"Successfully fetched and parsed Wikipedia text for authorship from URL: {url}")
    return text
def mixed_sentence_tokenize_for_authorship(text: str):
    # EN: Split text into sentences (supports Japanese and English)
    # JP: テキストを文ごとに分割（日本語・英語対応）
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]
# app.py の tokenize_mixed_for_authorship 関数を以下のように置き換える
def remove_punctuation_from_token(token: str) -> str:
    return ''.join(char for char in token if char not in PUNCTUATION_SET)
def preprocess_tokens_for_search(tokens_list: list[str]) -> list[str]:
    processed_tokens = [remove_punctuation_from_token(token) for token in tokens_list]
    return [token for token in processed_tokens if token]
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
        initial_tokens = word_tokenize(text)
    except Exception as e_tok_en:
        logging.error(f"NLTK word_tokenize for English text failed during authorship tokenization: {e_tok_en}")
        initial_tokens = text.split()
    punctuation_removed_tokens = preprocess_tokens_for_search(initial_tokens)
    lowercased_tokens = [token.lower() for token in punctuation_removed_tokens]
    return lowercased_tokens
    
def build_sentence_dataset_for_authorship(text: str, author_label: str, min_len: int = 30):
    # EN: Build a dataset of sentences and labels for authorship analysis
    # JP: 著者識別分析用の文とラベルのデータセットを作成
    sentences = mixed_sentence_tokenize_for_authorship(text)
    filtered = [s for s in sentences if len(s) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels

# === API Endpoints ===

@app.route('/')
def home():
    return "Hello, World! This is the home page."

@app.route('/api/search', methods=['POST'])
def kwic_search():
    data = request.json
    url = data.get('url', '').strip()
    query_input = data.get('query', '').strip() # ユーザーからの生の検索クエリ
    search_type = data.get('type', 'token').strip().lower()

    logging.info(f"Received KWIC search request. URL: {url}, Raw Query: '{query_input}', Type: {search_type}")

    # --- URLと基本的なクエリのバリデーション (変更なし) ---
    if not url:
        logging.warning("KWIC search request missing URL.")
        # メッセージは get_text_from_general_url に合わせて修正しても良い
        return jsonify({"error": "Please provide a Web Page URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme: {url}")
            return jsonify({"error": "Invalid URL scheme. Only http or https are allowed."}), 400
        if not parsed_url.hostname: # or is_restricted_hostname(parsed_url.hostname) from previous general URL suggestion
             logging.warning(f"Invalid or restricted URL hostname: {url}")
             return jsonify({"error": "Invalid or restricted URL hostname."}), 400
    except ValueError:
         logging.warning(f"Invalid URL format: {url}")
         return jsonify({"error": "Invalid URL format."}), 400
    except Exception as e:
         logging.error(f"Unexpected URL parsing error: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "Unexpected error during URL validation."}), 500

    if not query_input:
        logging.warning("KWIC search request missing query.")
        return jsonify({"error": "Please provide a search query."}), 400
    # --- ここまでバリデーション ---

    # --- ページからテキストを取得し、トークン化し、句読点除去 ---
    # この処理は全ての検索タイプで必要になる可能性があるため、分岐の前に行う
    words_from_page_original_case = []
    try:
        # get_text_from_url_for_kwic を使うか、汎用化された get_text_from_general_url を使うか選択
        # ここではユーザーが提供したコードに合わせて get_text_from_url_for_kwic を使用
        raw_text = get_text_from_url_for_kwic(url)
        text_cleaned = clean_text_for_kwic(raw_text)

        if not text_cleaned:
            logging.info(f"No text content extracted from URL: {url}")
            return jsonify({"results": [], "error": "Could not extract text content from the URL."}), 200

        initial_tokens_from_page = word_tokenize(text_cleaned)
        # これが検索対象となる、句読点除去済みのページのトークンリスト
        words_from_page_processed = preprocess_tokens_for_search(initial_tokens_from_page) 

        if not words_from_page_processed:
            logging.info(f"No searchable tokens after processing for URL: {url}")
            return jsonify({"results": [], "error": "No searchable words found in the URL after processing."}), 200

    except ValueError as e: # エラーメッセージを get_text_from_url_for_kwic から受け取る
        logging.warning(f"Error during URL processing for KWIC search {url}: {e}")
        return jsonify({"error": str(e)}), 400 # str(e) でエラーメッセージをそのまま返す
    except Exception as e:
        logging.error(f"Unexpected server error during URL/text processing: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during URL processing."}), 500
    # --- ここまでで words_from_page_processed が準備完了 ---

    results = [] # 最終的な検索結果リスト
    backend_context_window_size = 10 # これは変更なし
    target_tokens = []

    if search_type == 'token':
        # --- 検索クエリの準備 (句読点除去とバリデーション) ---
        raw_target_tokens = query_input.split()
        target_tokens_processed = preprocess_tokens_for_search(raw_target_tokens) # ★★★ target_tokens_processed をここで定義 ★★★

        if not target_tokens_processed:
            logging.warning(f"Token query '{query_input}' became empty after punctuation removal.")
            return jsonify({"error": "Query became empty after removing punctuation. Please provide a query with more than just punctuation."}), 400

        if not 1 <= len(target_tokens_processed) <= 5:
            logging.warning(f"Invalid token query length after punctuation removal: {target_tokens_processed} (Original: '{query_input}')")
            return jsonify({"error": "For token search, please enter one to five words (after punctuation removal)."}), 400
        # --- ここまで検索クエリ準備完了 ---

        # --- トークン検索とソートロジック (前回提案通り) ---
        words_from_page_lower = [w.lower() for w in words_from_page_processed]
        target_token_list_lower = [word.lower() for word in target_tokens_processed] # 正しく参照
        num_target_tokens = len(target_token_list_lower)

        temp_results_for_sorting = []
        all_observed_following_words = []

        for i in range(len(words_from_page_lower) - num_target_tokens + 1):
            if words_from_page_lower[i : i + num_target_tokens] == target_token_list_lower:
                before = words_from_page_processed[max(0, i - backend_context_window_size): i]
                matched_segment = words_from_page_processed[i : i + num_target_tokens]
                after = words_from_page_processed[i + num_target_tokens : i + num_target_tokens + backend_context_window_size]
                context_words_list = before + matched_segment + after

                current_following_word = ""
                following_word_index = i + num_target_tokens
                if following_word_index < len(words_from_page_processed):
                    current_following_word = words_from_page_processed[following_word_index].lower()
                    all_observed_following_words.append(current_following_word)

                temp_results_for_sorting.append({
                    "context_words": context_words_list,
                    "matched_start": len(before),
                    "matched_end": len(before) + num_target_tokens,
                    "original_text_index": i,
                    "raw_following_word": current_following_word
                })

        if not temp_results_for_sorting:
            logging.info(f"Query '{query_input}' (Type: token, Processed: '{' '.join(target_tokens_processed)}') not found.")
            # results は空のままなので、この後の共通処理で "not found" メッセージが返る
        else:
            overall_following_word_frequencies = Counter(all_observed_following_words)

            results_with_frequency_info = []
            for item in temp_results_for_sorting:
                item['overall_following_word_frequency'] = overall_following_word_frequencies.get(item['raw_following_word'], 0)
                results_with_frequency_info.append(item)

            results_with_frequency_info.sort(key=lambda x: (-x['overall_following_word_frequency'], x['original_text_index']))
            results = results_with_frequency_info # ソートされた結果を results に代入
        # --- トークン検索とソートロジックここまで ---

    elif search_type == 'pos':
        # --- POS検索ロジック (入力は words_from_page_processed を使用) ---
        target_pos_tag_query = query_input.strip().upper() # 句読点除去は不要、バリデーションは基本クエリバリデーションでカバー
        if not target_pos_tag_query: # query_inputが空かスペースのみの場合
             return jsonify({"error": f"For {search_type} search, please provide a valid tag."}), 400
        if " " in target_pos_tag_query: # スペースが含まれている場合
             return jsonify({"error": f"For {search_type} search, please enter a single tag (no spaces)."}), 400


        logging.info(f"--- Attempting POS tagging for query '{target_pos_tag_query}' ---")
        # ... (前回提案の nltk.pos_tag() の直接呼び出し、またはエラーハンドリング付きの明示的ロード)
        try:
            tagged_words = nltk.pos_tag(words_from_page_original_case)
        except Exception as e_pos_tag:
            logging.error(f"NLTK pos_tag failed: {e_pos_tag}\n{traceback.format_exc()}")
            return jsonify({"error": "Part-of-speech tagging failed on the server."}), 500

        target_pos_tag_query = query_input.upper()
        logging.info(f"POS tagging successful. Number of tagged words: {len(tagged_words)}")

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
                    "matched_start": len(before), "matched_end": len(before) + 1
                })
        # --- POS検索ロジックここまで ---

    elif search_type == 'entity':
        # --- Entity検索ロジック (入力は words_from_page_processed を使用) ---
        target_entity_type_query = query_input.upper()
        words_from_page_lower = [w.lower() for w in words_from_page_original_case]

        # Prefer spaCy if available for better accuracy
        spacy_ran = False
        if _SPACY_NLP:
            try:
                doc = _SPACY_NLP(" ".join(words_from_page_original_case))
                spacy_ran = True
                seen_token_spans = set()
                for ent in doc.ents:
                    if ent.label_.upper() == target_entity_type_query:
                        ent_tokens = word_tokenize(ent.text)
                        ent_tokens_lower = [t.lower() for t in ent_tokens]
                        n_ent = len(ent_tokens_lower)
                        # Find first matching slice in token list
                        for i in range(len(words_from_page_lower) - n_ent + 1):
                            if tuple(words_from_page_lower[i:i+n_ent]) == tuple(ent_tokens_lower):
                                if (i, i+n_ent) in seen_token_spans:
                                    break
                                seen_token_spans.add((i, i+n_ent))
                                before = words_from_page_original_case[max(0, i-backend_context_window_size):i]
                                after = words_from_page_original_case[i+n_ent:i+n_ent+backend_context_window_size]
                                context_words_list = before + words_from_page_original_case[i:i+n_ent] + after
                                results.append({
                                    "context_words": context_words_list,
                                    "matched_start": len(before),
                                    "matched_end": len(before)+n_ent
                                })
                                break
            except Exception as e_spa_entity:
                logging.error(f"spaCy entity extraction failed: {e_spa_entity}. Falling back to NLTK.\n{traceback.format_exc()}")
                spacy_ran = False

        if not spacy_ran:  # Fallback to original NLTK logic
            try:
                tagged_words_for_ner = nltk.pos_tag(words_from_page_original_case)
                chunked_entities_tree = nltk.ne_chunk(tagged_words_for_ner)
                iob_tags = nltk.chunk.util.tree2conlltags(chunked_entities_tree)
            except Exception as e_ner:
                logging.error(f"NLTK entity recognition failed: {e_ner}\n{traceback.format_exc()}")
                return jsonify({"error": "Entity recognition processing failed on the server."}), 500

            idx = 0
            while idx < len(iob_tags):
                word_original, _, iob_label = iob_tags[idx]
                if iob_label.startswith('B-') and iob_label[2:] == target_entity_type_query:
                    entity_words = [word_original]
                    start_idx = idx
                    next_idx = idx+1
                    while next_idx < len(iob_tags):
                        next_word, _, next_label = iob_tags[next_idx]
                        if next_label.startswith('I-') and next_label[2:] == target_entity_type_query:
                            entity_words.append(next_word)
                            next_idx += 1
                        else:
                            break
                    n_ent = len(entity_words)
                    before = words_from_page_original_case[max(0, start_idx-backend_context_window_size):start_idx]
                    after = words_from_page_original_case[next_idx:next_idx+backend_context_window_size]
                    context_words_list = before + entity_words + after
                    results.append({
                        "context_words": context_words_list,
                        "matched_start": len(before),
                        "matched_end": len(before)+n_ent
                    })
                    idx = next_idx
                else:
                    idx += 1
        if not target_entity_type_query:
             return jsonify({"error": f"For {search_type} search, please provide a valid entity type."}), 400
        if " " in target_entity_type_query:
             return jsonify({"error": f"For {search_type} search, please enter a single entity type (no spaces)."}), 400

    # --- 最終的なレスポンス ---
    if not results:
        logging.info(f"Query '{query_input}' (Type: {search_type}) not found in text from {url}")
        return jsonify({
            "results": [],
            "error": f"The query '{query_input}' (Type: {search_type}) was not found in the text from the provided URL."
        }), 200

    # Get the sort method from request, default to 'sequential'
    sort_method = request.json.get('sort_method', 'sequential')

    if sort_method == 'sequential':
        # Just return results in order found
        sorted_results = results

    elif sort_method == 'token':
        # Group results by the next token after match
        token_groups = {}
        for result in results:
            matched_end = result["matched_end"]
            context_words = result["context_words"]
            if matched_end < len(context_words):
                next_token = context_words[matched_end].lower()  # Just take the next single token
                if next_token not in token_groups:
                    token_groups[next_token] = []
                token_groups[next_token].append(result)

        # Sort by token frequency
        sorted_results = []
        token_freqs = [(token, len(group)) for token, group in token_groups.items()]
        for token, _ in sorted(token_freqs, key=lambda x: (-x[1], x[0])):  # Sort by freq desc, then token asc
            sorted_results.extend(token_groups[token])

    elif sort_method == 'pos':
        # Tag the results and group by POS of next word
        pos_groups = {}
        for result in results:
            matched_end = result["matched_end"]
            context_words = result["context_words"]
            if matched_end < len(context_words):
                next_words = context_words[matched_end:matched_end + 1]  # Get next word
                try:
                    tagged = nltk.pos_tag(next_words)
                    if tagged:
                        pos_tag = tagged[0][1]  # Get POS tag of next word
                        if pos_tag not in pos_groups:
                            pos_groups[pos_tag] = []
                        pos_groups[pos_tag].append(result)
                except Exception as e_pos:
                    logging.error(f"POS tagging failed for sorting: {e_pos}")
                    continue

        # Sort by POS tag frequency
        sorted_results = []
        pos_freqs = [(pos, len(group)) for pos, group in pos_groups.items()]
        for pos, _ in sorted(pos_freqs, key=lambda x: (-x[1], x[0])):  # Sort by freq desc, then POS tag asc
            sorted_results.extend(pos_groups[pos])

    else:
        # Invalid sort method, default to sequential
        sorted_results = results

    logging.info(f"Found {len(results)} occurrences for query '{query_input}' (Type: {search_type}) in text from {url}")
    return jsonify({"results": sorted_results})

@app.route('/api/authorship', methods=['POST'])
def authorship_analysis(): # No changes to this function
    # EN: Authorship analysis API endpoint
    # JP: 著者識別分析APIエンドポイント
    data = request.json
    url_a = data.get('url_a', '').strip()
    url_b = data.get('url_b', '').strip()

    logging.info(f"Received authorship analysis request for URL A: {url_a}, URL B: {url_b}")


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
