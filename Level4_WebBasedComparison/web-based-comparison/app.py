# app.py

# --- Imports ---
from collections import Counter
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk 
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
import string
import os
import spacy

# --- グローバル変数と初期化設定 ---
_SPACY_NLP = None
# RenderのRoot Directoryを基準としたNLTKデータディレクトリの期待パス
_APP_ROOT_ON_RENDER = '/opt/render/project/src/Level4_WebBasedComparison/web-based-comparison' # RenderのRoot Directory設定に合わせる
_NLTK_DATA_DIR_IN_REPO = 'nltk_data_on_render' # アプリケーションルート直下のフォルダ名
_EXPECTED_NLTK_DATA_PATH = os.path.join(_APP_ROOT_ON_RENDER, _NLTK_DATA_DIR_IN_REPO)

# 環境変数NLTK_DATAを最優先、次に上記で構成したパス、最終フォールバックはnltkデフォルト
_NLTK_DATA_DIR = os.environ.get('NLTK_DATA', _EXPECTED_NLTK_DATA_PATH)


def initialize_nlp_resources():
    global _SPACY_NLP
    global _NLTK_DATA_DIR # この関数内でグローバル変数を参照

    # spaCyモデルのロード
    print("--- [APP STARTUP] Attempting to load spaCy model 'en_core_web_sm'... ---")
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
        print("--- [APP STARTUP] spaCy model 'en_core_web_sm' loaded successfully. ---")
    except OSError as e:
        print(f"--- [APP STARTUP] CRITICAL ERROR: spaCy model 'en_core_web_sm' not found or failed to load: {e} ---")
        print("--- [APP STARTUP] Ensure 'python -m spacy download en_core_web_sm' was run in Build Command. ---")
        sys.exit(1)
    except Exception as e_spacy_unexpected:
        print(f"--- [APP STARTUP] An unexpected error occurred while loading spaCy model: {e_spacy_unexpected} ---")
        sys.exit(1)

    # NLTKデータパスの設定と確認
    print(f"--- [APP STARTUP PRE-CHECK] Effective NLTK_DATA directory to be used: {_NLTK_DATA_DIR} ---")
    if not os.path.isdir(_NLTK_DATA_DIR):
        print(f"--- [APP STARTUP PRE-CHECK] CRITICAL ERROR: NLTK data directory '{_NLTK_DATA_DIR}' does NOT exist. ---")
        print(f"--- [APP STARTUP PRE-CHECK] This directory should have been created by 'download_nltk_for_authorship.py' during build. ---")
        sys.exit(1)
    else:
        print(f"--- [APP STARTUP PRE-CHECK] NLTK data directory '{_NLTK_DATA_DIR}' exists. ---")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout # Renderはstdout/stderrのログを収集するのでこれでOK
)

# アプリケーション起動時にNLPリソースを初期化
initialize_nlp_resources()

# --- Authorship Attribution Imports (initialize_nlp_resourcesの後) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score
# --- Multilingual Support Libraries ---
from langdetect import detect as lang_detect
import fugashi
from stopwordsiso import stopwords

PUNCTUATION_SET = set(string.punctuation + '。、「」』『【】・（）　')

# --- Global Settings for Authorship Attribution ---
_TAGGER = None
try:
    _TAGGER = fugashi.Tagger()
    logging.info("fugashi Tagger initialized successfully.")
except Exception as e_fugashi:
    logging.error(f"Failed to initialize fugashi Tagger: {e_fugashi}. Japanese tokenization for authorship will be limited.")

_EN_SW = set()
_JA_SW = set()
try:
    if nltk.data.find('corpora/words'): # wordsコーパスが利用可能か確認
        _EN_SW = stopwords("en")
    if nltk.data.find('corpora/words'): # 日本語ストップワードもwordsに依存する場合がある(stopwordsisoの実装による)
        _JA_SW = stopwords("ja")
    if _EN_SW or _JA_SW:
        logging.info("English and/or Japanese stopwords (for authorship) loaded.")
    else:
        logging.warning("Could not load stopwords for authorship.")
except Exception as e_stopwords:
    logging.error(f"Failed to load stopwords (for authorship): {e_stopwords}")

_ALL_SW = sorted(list(_EN_SW.union(_JA_SW)))
_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# --- Flask App Setup ---
app = Flask(__name__)
FRONTEND_GITHUB_PAGES_ORIGIN = "https://analphy.github.io"
FRONTEND_DEV_ORIGIN_3000 = "http://localhost:3000"
FRONTEND_DEV_ORIGIN_8080 = "http://localhost:8080"
allowed_origins_list = [
    FRONTEND_GITHUB_PAGES_ORIGIN,
    FRONTEND_DEV_ORIGIN_3000,
    FRONTEND_DEV_ORIGIN_8080,
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins_list}})
print("--- [APP STARTUP] Flask application initialized and CORS configured. ---")
logging.info("Application setup complete. Ready for requests.")


# === Text processing for KWIC Search ===
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
        for tag_to_remove in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                        'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                        'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag_to_remove.decompose()
        content = soup.find(id='mw-content-text')
        if content:
            wiki_elements_to_remove = ['toc', 'reference', 'reflist', 'navbox', 'metadata', 'catlinks', 
                           'mw-editsection', 'mw-references', 'mw-navigation', 'mw-footer',
                           'sistersitebox', 'noprint', 'mw-jump-to-nav', 'mw-indicator',
                           'mw-wiki-logo', 'mw-page-tools', 'printfooter', 'mw-revision']
            for element_to_remove in content.find_all(['div', 'section', 'span', 'nav', 'footer']): # Added span, nav, footer
                if any(cls in (element_to_remove.get('class', []) or []) for cls in wiki_elements_to_remove):
                    element_to_remove.decompose()
            text = content.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched and parsed URL (for KWIC search): {url}")
        return text
    except (URLError, HTTPError) as e_url:
        logging.error(f"URL Error fetching (KWIC search) {url}: {e_url.reason}")
        raise ValueError(f"Could not access the URL. Reason: {e_url.reason}")
    except TimeoutError:
         logging.error(f"Timeout fetching (KWIC search) {url} after 15 seconds.")
         raise ValueError("URL fetching timed out.")
    except ValueError as e_val: # Specific ValueError from above
          logging.error(f"Value Error processing (KWIC search) {url}: {e_val}")
          raise
    except Exception as e_gen:
        logging.error(f"An unexpected error in get_text_from_url_for_kwic for {url}: {e_gen}\n{traceback.format_exc()}")
        raise ValueError("An unexpected error occurred while fetching or processing the URL.")

def clean_text_for_kwic(text):
    text = re.sub(r'Jump to.*?content', '', text, flags=re.IGNORECASE)
    text = re.sub(r'From Wikipedia.*?encyclopedia', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Categories\s*:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hidden categories\s*:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]', '', text)  # citations like [1], [23]
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE) # Specific [edit] links
    text = re.sub(r'Edit.*?section', '', text, flags=re.IGNORECASE)
    text = re.sub(r'oldid=\d+', '', text)
    text = re.sub(r'\s*\(\s*disambiguation\s*\)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CS1.*?maintenance.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'Articles.*?containing.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'Use mdy dates.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'Short description.*?\n', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'Category:.*?(?=\n|$)', '', text)
    text = re.sub(r'Coordinates:.*?(?=\n|$)', '', text) # Coordinates
    # Further specific cleanup if needed
    text = re.sub(r'Retrieved from.*?(?=\n|$)',"",text, flags=re.IGNORECASE)
    text = re.sub(r'Navigation menu', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Personal tools', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Not logged in', '', text, flags=re.IGNORECASE)
    text = re.sub(r'TalkContributionsCreate accountLog in', '', text, flags=re.IGNORECASE)
    text = re.sub(r'NamespacesArticleTalk', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ViewsReadEditView history', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Search Wikipedia', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Main pageContentsCurrent eventsRandom articleAbout WikipediaContact usDonate', '', text, flags=re.IGNORECASE)
    text = re.sub(r'HelpLearn to editCommunity portalRecent changesUpload file', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ToolsWhat links hereRelated changesSpecial pagesPermanent linkPage informationCite this pageWikidata item', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Print/exportDownload as PDFPrintable version', '', text, flags=re.IGNORECASE)
    text = re.sub(r'In other projectsWikimedia Commons', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Languages‪Deutsch‬Español한국어ItalianoРусскийTiếng ViệtПравить ссылки', '', text, flags=re.IGNORECASE) # Example for other languages links
    text = re.sub(r'Privacy policyAbout WikipediaDisclaimersContact WikipediaCode of ConductDevelopersStatisticsCookie statementMobile view', '', text, flags=re.IGNORECASE)
    # Final cleanup of multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_punctuation_from_token(token: str) -> str:
    return ''.join(char for char in token if char not in PUNCTUATION_SET)

# preprocess_tokens_for_search はKWIC検索のトークン検索クエリの前処理にも使われる
def preprocess_tokens_for_search(tokens_list: list[str]) -> list[str]:
    processed_tokens = [remove_punctuation_from_token(token) for token in tokens_list]
    return [token for token in processed_tokens if token] # 空になったトークンを除去

# === Authorship Attribution Helpers ===
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
                 raise ValueError("The provided URL does not appear to be an HTML page.")
    except (URLError, HTTPError) as e_url_auth:
        logging.error(f"URL Error fetching (authorship) {url}: {e_url_auth.reason}")
        raise ValueError(f"Could not access the URL for authorship. Reason: {e_url_auth.reason}")
    except TimeoutError:
        logging.error(f"Timeout fetching (authorship) {url}")
        raise ValueError("URL fetching for authorship timed out.")
    except Exception as err_fetch_auth:
        logging.error(f"Error fetching URL (authorship) {url}: {err_fetch_auth}\n{traceback.format_exc()}")
        raise ValueError("An unexpected error occurred while fetching URL for authorship.")
    
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                     'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                     'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
        tag.decompose()
    content = soup.find(id='mw-content-text')
    if content:
        for section in content.find_all(['div', 'section'], class_=['toc', 'reference', 'reflist', 'navbox', 'metadata', 'catlinks']):
            section.decompose()
        for element in content.find_all(['span', 'div'], class_=['mw-editsection', 'reference', 'reflist']): # Consider removing more specific classes
            element.decompose()
        text = content.get_text(separator=' ')
    else:
        text = soup.get_text(separator=' ') # Fallback
    
    # More aggressive cleaning for authorship
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)  # Citations like [1], [2,3]
    text = re.sub(r'\[.*?\]', '', text) # Any other content in square brackets
    text = re.sub(r'\(listen\)', '', text, flags=re.IGNORECASE) # (listen)
    text = re.sub(r'Edit.*?section', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Jump to.*?content', '', text, flags=re.IGNORECASE)
    text = re.sub(r'From Wikipedia.*?encyclopedia', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Categories\s*:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    logging.info(f"Successfully fetched and parsed Wikipedia text for authorship from URL: {url}")
    return text

def mixed_sentence_tokenize_for_authorship(text: str):
    # NLTKの文分割に依存しないように修正 (正規表現ベース)
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def tokenize_mixed_for_authorship(text: str):
    # 著者分析ではNLTKのword_tokenizeを使用 (Build Commandでpunktがダウンロードされている前提)
    try:
        initial_tokens = word_tokenize(text)
    except LookupError: # punkt が見つからない場合へのフォールバック
        logging.warning("NLTK 'punkt' resource not found for authorship tokenization, falling back to simple split.")
        initial_tokens = text.split()
    except Exception as e_tok_auth:
        logging.error(f"NLTK word_tokenize failed during authorship tokenization: {e_tok_auth}")
        initial_tokens = text.split()
    punctuation_removed_tokens = preprocess_tokens_for_search(initial_tokens)
    lowercased_tokens = [token.lower() for token in punctuation_removed_tokens]
    return lowercased_tokens
    
def build_sentence_dataset_for_authorship(text: str, author_label: str, min_len: int = 30):
    sentences = mixed_sentence_tokenize_for_authorship(text)
    filtered = [s for s in sentences if len(s) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels

# === API Endpoints ===

@app.route('/')
def home():
    return "Authorship Analysis API is running!"

@app.route('/api/search', methods=['POST'])
def kwic_search():
    data = request.json
    url = data.get('url', '').strip()
    query_input = data.get('query', '').strip()
    search_type = data.get('type', 'token').strip().lower()
    sort_method = data.get('sort_method', 'sequential')

    logging.info(f"Received KWIC search. URL: {url}, Query: '{query_input}', Type: {search_type}, Sort: {sort_method}")

    if not url: return jsonify({"error": "URL is required."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https'] or not parsed_url.hostname:
            return jsonify({"error": "Invalid URL (must be http or https with a hostname)."}), 400
    except Exception: return jsonify({"error": "Invalid URL format."}), 400
    if not query_input: return jsonify({"error": "Search query is required."}), 400

    try:
        raw_text = get_text_from_url_for_kwic(url)
        text_content = clean_text_for_kwic(raw_text)
        if not text_content:
            return jsonify({"results": [], "error": "No text content extracted."}), 200
    except ValueError as e: return jsonify({"error": str(e)}), 400
    except Exception as e_proc:
        logging.error(f"KWIC URL processing error: {e_proc}\n{traceback.format_exc()}")
        return jsonify({"error": "Server error during URL processing."}), 500

    # app.py の kwic_search 関数内

# ... (関数の冒頭、doc_for_kwic の取得までは変更なし) ...
# overall_word_frequencies の計算は、ページ全体の頻度が必要な場合に備えて残しても良いですが、
# 今回のPOS検索のソートでは「検索結果内での頻度」を使うため、直接は使用しません。

    results = [] # 最終結果を格納するリストを初期化
    backend_context_window_size = 10

    if not _SPACY_NLP:
        logging.error("--- KWIC SEARCH ERROR: spaCy NLP model (_SPACY_NLP) is None! ---")
        return jsonify({"error": "NLP model is not available. Cannot process search."}), 500
    try:
        doc_for_kwic = _SPACY_NLP(text_content) # text_content は事前に取得・クリーン済みとします
    except Exception as e_spa_proc:
        logging.error(f"spaCy processing failed for KWIC: {e_spa_proc}\n{traceback.format_exc()}")
        return jsonify({"error": "Text processing (spaCy) failed."}), 500


    if search_type == 'token':
        # --- Token検索のロジック (前回の後続単語1つの頻度ソートを維持) ---
        raw_target_tokens = query_input.split()
        target_tokens_processed_lower = [remove_punctuation_from_token(word).lower() for word in raw_target_tokens if remove_punctuation_from_token(word)]
        if not target_tokens_processed_lower or not (1 <= len(target_tokens_processed_lower) <= 5):
            return jsonify({"error": "Token query must be 1-5 words after punctuation removal."}), 400
        
        num_target_tokens = len(target_tokens_processed_lower)
        all_observed_following_words_for_token_sort = [] 
        
        # token検索では、まずマッチ箇所を results に追加
        for i in range(len(doc_for_kwic) - num_target_tokens + 1):
            candidate_doc_tokens = [doc_for_kwic[j] for j in range(i, i + num_target_tokens)]
            candidate_processed_lower = [
                token_text_lower for tok in candidate_doc_tokens 
                if (token_text_lower := remove_punctuation_from_token(tok.text).lower())
            ]
            if len(candidate_processed_lower) == num_target_tokens and candidate_processed_lower == target_tokens_processed_lower:
                start_context_idx = max(0, i - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), i + num_target_tokens + backend_context_window_size)
                context_words_list = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                matched_start_in_context = i - start_context_idx
                matched_end_in_context = matched_start_in_context + num_target_tokens
                current_following_word_1 = ""
                following_token_idx_in_doc = i + num_target_tokens
                if following_token_idx_in_doc < len(doc_for_kwic):
                    current_following_word_1 = doc_for_kwic[following_token_idx_in_doc].text.lower()
                    processed_fw1 = remove_punctuation_from_token(current_following_word_1)
                    if processed_fw1:
                         all_observed_following_words_for_token_sort.append(processed_fw1)
                results.append({
                    "context_words": context_words_list,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_end_in_context,
                    "original_text_index": i, 
                    "raw_following_word": remove_punctuation_from_token(current_following_word_1) # 正規化して保持
                })
        
        if results: # token検索の結果があれば、後続単語1つの頻度でソート
            logging.info(f"Token search found {len(results)} matches. Sorting by 1st following word frequency.")
            current_overall_following_word_frequencies = Counter(all_observed_following_words_for_token_sort)
            for item in results:
                item['overall_following_word_frequency'] = current_overall_following_word_frequencies.get(item['raw_following_word'], 0)
            results.sort(key=lambda x: (-x.get('overall_following_word_frequency', 0), x.get('original_text_index', 0)))
        # --- Token検索のロジックここまで ---

   # POS検索のソート処理を修正
# app.py の該当部分（行番号約200-250あたり）を以下に置き換え

    elif search_type == 'pos':
        target_pos_tag_query = query_input.strip().upper()
    if not target_pos_tag_query or " " in target_pos_tag_query:
         return jsonify({"error": "For POS search, enter a single valid tag."}), 400

    raw_results_for_pos_sort = [] 
    logging.info(f"--- [POS SEARCH] Collecting raw results for POS tag '{target_pos_tag_query}' ---")
    for i, token in enumerate(doc_for_kwic):
        if token.tag_ == target_pos_tag_query: # マッチする品詞タグ
            start_context_idx = max(0, i - backend_context_window_size)
            end_context_idx = min(len(doc_for_kwic), i + 1 + backend_context_window_size)
            context_doc_tokens = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
            matched_start_in_context = i - start_context_idx
            
            matched_word_text_original = token.text
            matched_word_text_processed = remove_punctuation_from_token(matched_word_text_original).lower()
            
            following_word_1_processed = ""
            if i + 1 < len(doc_for_kwic):
                fw1_raw = doc_for_kwic[i + 1].text
                following_word_1_processed = remove_punctuation_from_token(fw1_raw).lower()
            
            raw_results_for_pos_sort.append({
                "context_words": context_doc_tokens,
                "matched_start": matched_start_in_context,
                "matched_end": matched_start_in_context + 1,
                "original_text_index": i,
                "matched_word_text_original": matched_word_text_original,
                "matched_word_text_processed": matched_word_text_processed,
                "following_word_1_processed": following_word_1_processed
            })
    
    if raw_results_for_pos_sort:
        # 1. マッチした単語の頻度を計算
        matched_word_texts = [item['matched_word_text_processed'] for item in raw_results_for_pos_sort if item['matched_word_text_processed']]
        matched_word_frequencies = Counter(matched_word_texts)
        logging.info(f"--- [POS SEARCH] Frequencies of matched POS words in results: {matched_word_frequencies.most_common()} ---")
        
        for item in raw_results_for_pos_sort:
            item['matched_pos_word_freq_in_results'] = matched_word_frequencies.get(item['matched_word_text_processed'], 0)

        # 2. 後続単語の頻度も計算（補助的なソートキーとして使用）
        all_following_words_in_pos_results = [item['following_word_1_processed'] for item in raw_results_for_pos_sort if item['following_word_1_processed']]
        following_word_freq_in_pos_results = Counter(all_following_words_in_pos_results)
        logging.info(f"--- [POS SEARCH] Frequencies of 1st following words in POS results (Top 5): {following_word_freq_in_pos_results.most_common(5)} ---")
        
        for item in raw_results_for_pos_sort:
            item['following_word_1_freq_in_results'] = following_word_freq_in_pos_results.get(item['following_word_1_processed'], 0)
        
        # 3. ★★★ 修正されたソート処理 ★★★
        # 頻度グループごとに処理し、同じ頻度内では出現順を保持
        frequency_groups = {}
        for item in raw_results_for_pos_sort:
            freq = item.get('matched_pos_word_freq_in_results', 0)
            if freq not in frequency_groups:
                frequency_groups[freq] = []
            frequency_groups[freq].append(item)
        
        # 各頻度グループ内を出現順でソート
        for freq in frequency_groups:
            frequency_groups[freq].sort(key=lambda x: x.get('original_text_index', 0))
        
        # 頻度の高い順にグループを結合
        sorted_results = []
        for freq in sorted(frequency_groups.keys(), reverse=True):
            sorted_results.extend(frequency_groups[freq])
        
        results = sorted_results
        
        # デバッグログ（頻度別の結果表示）
        logging.info("--- [POS SEARCH] Results grouped by frequency ---")
        for freq in sorted(frequency_groups.keys(), reverse=True):
            words_in_group = [item['matched_word_text_processed'] for item in frequency_groups[freq]]
            unique_words = list(dict.fromkeys(words_in_group))  # 順序を保持しつつ重複除去
            logging.info(f"  Frequency {freq}: {unique_words} ({len(frequency_groups[freq])} occurrences)")
            
        # ソート後の最初の数件を確認
        logging.info("--- [POS SEARCH] After sorting (first 10 items) ---")
        for i_debug, item_debug in enumerate(results[:10]):
            logging.info(f"  Item {i_debug}: Word='{item_debug['matched_word_text_processed']}'(freq={item_debug.get('matched_pos_word_freq_in_results','N/A')}), "
                         f"OrigIdx={item_debug['original_text_index']}")
    else:
        results = []
        # --- POS検索のロジックここまで ---

    elif search_type == 'entity':
        # ★★★ Entity検索のロジックとソートを修正 ★★★
        target_entity_type_query = query_input.strip().upper()
        if not target_entity_type_query or " " in target_entity_type_query:
            return jsonify({"error": "For Entity search, enter a single valid entity type."}), 400
        
        # 1. マッチするアイテムを全て収集し、必要な情報を付加
        raw_results_for_entity_sort = [] 
        logging.info(f"--- [ENTITY SEARCH] Collecting raw results for type '{target_entity_type_query}' ---")
        for ent_idx, ent in enumerate(doc_for_kwic.ents): # doc_for_kwic は処理済みのspaCy Docオブジェクト
            if ent.label_.upper() == target_entity_type_query:
                start_token_idx = ent.start
                end_token_idx = ent.end 
                item_match_len = end_token_idx - start_token_idx

                start_context_idx = max(0, start_token_idx - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), end_token_idx + backend_context_window_size)
                context_doc_tokens = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                matched_start_in_context = start_token_idx - start_context_idx
                
                matched_entity_text_original = ent.text
                # マッチしたエンティティ自体を正規化 (小文字化、句読点除去)
                matched_entity_text_processed = remove_punctuation_from_token(matched_entity_text_original).lower()
                
                # (オプション) 直後の単語の情報も収集しておく (もし第2ソートキーで使いたければ)
                following_word_1_processed = ""
                idx_after_entity = ent.end 
                if idx_after_entity < len(doc_for_kwic):
                    fw1_raw = doc_for_kwic[idx_after_entity].text
                    following_word_1_processed = remove_punctuation_from_token(fw1_raw).lower()

                raw_results_for_entity_sort.append({
                    "context_words": context_doc_tokens,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_start_in_context + item_match_len,
                    "original_text_index": start_token_idx, 
                    "matched_entity_text_original": matched_entity_text_original,
                    "matched_entity_text_processed": matched_entity_text_processed,
                    "following_word_1_processed": following_word_1_processed 
                })
        
        logging.info(f"--- [ENTITY SEARCH] Collected {len(raw_results_for_entity_sort)} raw entity matches. ---")
        # デバッグ用に収集したエンティティの一部を表示
        # for i_debug, item_debug in enumerate(raw_results_for_entity_sort[:5]):
        #    logging.info(f"  Raw Entity Item {i_debug}: Text='{item_debug['matched_entity_text_processed']}', OrigIdx={item_debug['original_text_index']}")

        if raw_results_for_entity_sort:
            # 2. マッチしたエンティティ自体の「検索結果内での」出現頻度を計算
            entity_texts_in_results = [item['matched_entity_text_processed'] for item in raw_results_for_entity_sort if item['matched_entity_text_processed']]
            entity_frequencies_in_results = Counter(entity_texts_in_results)
            logging.info(f"--- [ENTITY SEARCH] Frequencies of matched entities in results (Top 5): {entity_frequencies_in_results.most_common(5)} ---")
            
            for item in raw_results_for_entity_sort:
                item['matched_entity_freq_in_results'] = entity_frequencies_in_results.get(item['matched_entity_text_processed'], 0)

            # (オプション: もし後続単語頻度も第二キーにしたい場合はここで計算・追加)
            # all_following_words_in_entity_results = [item['following_word_1_processed'] for item in raw_results_for_entity_sort if item['following_word_1_processed']]
            # following_word_freq_in_entity_results = Counter(all_following_words_in_entity_results)
            # for item in raw_results_for_entity_sort:
            #     item['following_word_1_freq_in_results'] = following_word_freq_in_entity_results.get(item['following_word_1_processed'], 0)

            # 3. ソート実行
            #    キー1: マッチしたエンティティ自体の検索結果内頻度 (降順)
            #    キー2: 元のテキストでの出現位置 (昇順)
            #    (もし後続単語頻度もキーにするなら、ソートキーのタプルに追加)
            logging.info("--- [ENTITY SEARCH] Sorting results by 'matched_entity_freq_in_results' (desc) then 'original_text_index' (asc)... ---")
            raw_results_for_entity_sort.sort(key=lambda x: (
                -x.get('matched_entity_freq_in_results', 0), # 最優先キー
                # -x.get('following_word_1_freq_in_results', 0), # ← もし後続単語頻度もキーにする場合
                x.get('original_text_index', 0)                # 第二（または第三）優先キー
            ))
            results = raw_results_for_entity_sort # ソートされた結果を最終結果に
            
            # デバッグログ (ソート後の最初の数件)
            logging.info("--- [ENTITY SEARCH] After sorting (first 5 items with freq) ---")
            for i_debug, item_debug in enumerate(results[:5]):
                logging.info(f"  Item {i_debug}: Entity='{item_debug['matched_entity_text_processed']}' (Freq={item_debug.get('matched_entity_freq_in_results', 'N/A')}), "
                             # f"NextWord='{item_debug['following_word_1_processed']}' (Freq={item_debug.get('following_word_1_freq_in_results', 'N/A')}), " # 後続単語頻度もキーにした場合
                             f"OrigIdx={item_debug['original_text_index']}")
        else:
            results = [] # マッチがなければ空
        # --- Entity検索のロジックここまで ---

    # --- 共通の処理 (結果がない場合) ---
    if not results:
        return jsonify({"results": [], "error": f"Query '{query_input}' (Type: {search_type}) not found."}), 200

    # --- フロントエンドからの sort_method に基づく追加のソート (オプション) ---
    # sorted_results は、この時点で各検索タイプごとのデフォルトソートが適用されている
    # (token検索は後続単語1頻度、pos検索はNNP頻度→後続単語1頻度、entity検索はエンティティ頻度)
    final_sorted_results = results 

    # もし、フロントエンドから明示的に 'pos' ソート (後続単語の「品詞タグ」頻度) が要求され、
    # かつ、現在の検索タイプが 'entity' の場合（まだこのソートが適用されていない場合）にのみ実行
    if sort_method == 'pos' and search_type == 'entity': 
        logging.info(f"--- [ENTITY SEARCH] Applying 'following word POS tag frequency' sort as per sort_method='pos' for entity results. ---")
        pos_groups_for_entity_alt_sort = {}
        for res_item in results: # results は Entity検索でエンティティ自体の頻度でソートされたもの
            item_start_idx_in_doc = res_item.get("original_text_index")
            item_match_len_val = res_item["matched_end"] - res_item["matched_start"]
            if item_start_idx_in_doc is not None:
                next_token_global_idx = item_start_idx_in_doc + item_match_len_val
                if next_token_global_idx < len(doc_for_kwic):
                    next_word_token_obj = doc_for_kwic[next_token_global_idx]
                    pos_tag = next_word_token_obj.tag_ # 品詞タグを取得
                    pos_groups_for_entity_alt_sort.setdefault(pos_tag, []).append(res_item)
        if pos_groups_for_entity_alt_sort:
            temp_list_entity_alt_sort = []
            for tag, group_items in sorted(pos_groups_for_entity_alt_sort.items(), key=lambda x: (-len(x[1]), x[0])):
                temp_list_entity_alt_sort.extend(sorted(group_items, key=lambda x: x.get("original_text_index", 0)))
            final_sorted_results = temp_list_entity_alt_sort
    elif sort_method == 'sequential' and search_type != 'token': # Token以外で出現順に戻したい場合
        logging.info(f"--- Sorting '{search_type}' results by original text index as per sort_method='sequential'. ---")
        final_sorted_results.sort(key=lambda x: x.get('original_text_index', 0))


    logging.info(f"KWIC search successful. Returning {len(final_sorted_results)} results for type '{search_type}' with sort '{sort_method}'.")
    return jsonify({"results": final_sorted_results})

@app.route('/api/authorship', methods=['POST'])
def authorship_analysis():
    data = request.json
    url_a = data.get('url_a', '').strip()
    url_b = data.get('url_b', '').strip()
    logging.info(f"Received authorship analysis request for URL A: {url_a}, URL B: {url_b}")

    for i, url_val in enumerate([url_a, url_b]):
        label = "A" if i == 0 else "B"
        try:
            parsed = urlparse(url_val)
            if not parsed.scheme in ['http', 'https'] or not parsed.hostname or not (parsed.hostname.lower() == 'wikipedia.org' or parsed.hostname.lower().endswith('.wikipedia.org')):
                return jsonify({"error": f"URL for Author {label} ('{url_val}') must be a valid Wikipedia URL."}), 400
        except ValueError:
            return jsonify({"error": f"Invalid URL format for Author {label}: {url_val}"}), 400
    try:
        logging.info("Fetching Wikipedia pages for authorship...")
        text_a = fetch_wikipedia_text_for_authorship(url_a)
        text_b = fetch_wikipedia_text_for_authorship(url_b)
        if not text_a or not text_b:
            return jsonify({"error": "Failed to fetch content from one or both Wikipedia pages."}), 500
        
        logging.info("Building sentence dataset for authorship...")
        sentences_a, labels_a = build_sentence_dataset_for_authorship(text_a, "AuthorA")
        sentences_b, labels_b = build_sentence_dataset_for_authorship(text_b, "AuthorB")
        
        min_samples_per_class = 2 # train_test_splitのstratifyに最低限必要なサンプル数
        if not sentences_a or not sentences_b or len(sentences_a) < min_samples_per_class or len(sentences_b) < min_samples_per_class :
            return jsonify({"error": "Could not extract enough valid sentences (min 30 chars, min 2 per author) for analysis."}), 400

        all_sentences = sentences_a + sentences_b
        all_labels = labels_a + labels_b
        
        if len(all_sentences) < 5: # 全体で最低5サンプルは欲しい
             return jsonify({"error": "Too few sentences overall (min 5) for reliable model training."}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        logging.info(f"Authorship Training samples: {len(X_train)} — Test samples: {len(X_test)}")

        if not X_train or not X_test: # 分割後空にならないか確認
            return jsonify({"error": "Failed to create training/testing sets. Likely insufficient data."}), 400

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
            top_indices = log_probs.argsort()[-10:][::-1] # 上位10単語
            top_terms = [feature_names[i] for i in top_indices]
            distinctive_words[author] = top_terms
            
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        report_target_names = sorted(list(clf.classes_)) # クラス名をソートして渡す
        report = sk_classification_report(y_test, preds, digits=3, zero_division=0, target_names=report_target_names)
        
        sample_predictions = []
        num_samples_to_show = min(len(X_test), 5)
        for i in range(num_samples_to_show):
            sent = X_test[i]
            true_label = y_test[i]
            pred_label = preds[i]
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
    except ValueError as ve_auth: # Renamed variable
        logging.error(f"ValueError during authorship analysis: {ve_auth}\n{traceback.format_exc()}")
        return jsonify({"error": str(ve_auth)}), 400
    except Exception as e_auth_unexpected: # Renamed variable
        logging.error(f"An unexpected error during authorship analysis: {e_auth_unexpected}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during authorship analysis."}), 500

if __name__ == "__main__":
    # このブロックはRender環境では通常実行されない (gunicornが直接 `app` オブジェクトをロードするため)
    # ローカルでのテスト実行用
    print("--- Flask development server starting via __main__ (for local testing) ---")
    # initialize_nlp_resources() は既にグローバルスコープで呼び出されているはず
    
    if not _SPACY_NLP: # 再確認
        print("CRITICAL (from __main__): spaCy model could not be loaded. The application might not work correctly if run directly.")
        # sys.exit(1) # ローカルテストなので、必ずしも終了させなくても良いかもしれない
    
    if not _TAGGER:
        print("Warning (from __main__): Fugashi Tagger failed to initialize. Japanese NLP features for authorship attribution will be limited.")
    
    print("Starting Flask app server with app.run()...")
    # RenderはPORT環境変数を設定するので、ローカル用にデフォルトポートを指定
    local_port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, port=local_port, host='0.0.0.0')