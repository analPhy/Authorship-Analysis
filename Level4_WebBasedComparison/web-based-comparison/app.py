# app.py

# --- Imports ---
from collections import Counter, defaultdict
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
    # sort_method はユーザーコードにあったので残しますが、今回の要件に合わせて主に頻度ソートを実装します
    sort_method_from_request = data.get('sort_method', 'frequency_then_sequential') # デフォルトを頻度順に変更

    logging.info(f"Received KWIC search. URL: {url}, Query: '{query_input}', Type: {search_type}, Sort: {sort_method_from_request}")

    # ... (URLと基本クエリのバリデーションはユーザーコードのものを尊重) ...
    if not url: return jsonify({"error": "URL is required."}), 400
    # ... (URLパースとホスト名チェック - SSRF対策も考慮すべき) ...
    if not query_input: return jsonify({"error": "Search query is required."}), 400


    try:
        raw_text = get_text_from_url_for_kwic(url) # ユーザー定義の関数を使用
        text_content = clean_text_for_kwic(raw_text) # ユーザー定義の関数を使用
        if not text_content:
            return jsonify({"results": [], "error": "No text content extracted."}), 200
    except ValueError as e: return jsonify({"error": str(e)}), 400
    except Exception as e_proc:
        logging.error(f"KWIC URL processing error: {e_proc}\n{traceback.format_exc()}")
        return jsonify({"error": "Server error during URL processing."}), 500

    if not _SPACY_NLP: # spaCyモデルのチェック
        logging.error("--- KWIC SEARCH ERROR: spaCy NLP model (_SPACY_NLP) is None! ---")
        return jsonify({"error": "NLP model is not available. Cannot process search."}), 500
    try:
        doc_for_kwic = _SPACY_NLP(text_content)
    except Exception as e_spa_proc:
        logging.error(f"spaCy processing failed for KWIC: {e_spa_proc}\n{traceback.format_exc()}")
        return jsonify({"error": "Text processing (spaCy) failed."}), 500

    results_with_metadata = [] # ソート用メタデータを含む一時リスト
    backend_context_window_size = 10

# app.py (kwic_search 関数の search_type == 'token' 部分の修正)

# ... (既存のimportや関数定義、kwic_search関数の冒頭部分は変更なし) ...
# ... (text_content の取得、doc_for_kwic = _SPACY_NLP(text_content) の処理までは変更なし) ...

    results_with_metadata = [] # ソート後の最終結果を格納するリスト
    backend_context_window_size = 10

    if search_type == 'token':
        # --- Token検索のロジック (次の単語の頻度でソート) ---
        raw_target_tokens = query_input.split()
        target_tokens_processed_lower = [
            token_lower for token in raw_target_tokens 
            if (token_lower := remove_punctuation_from_token(token).lower())
        ]

        if not target_tokens_processed_lower or not (1 <= len(target_tokens_processed_lower) <= 5):
            return jsonify({"error": "Token query must be 1-5 words after punctuation removal and processing."}), 400
        
        num_target_tokens = len(target_tokens_processed_lower)
        
        temp_token_matches = [] # マッチ情報とソート用メタデータを一時的に格納
        
        # spaCyのDocオブジェクトの長さを正しく取得するために len(doc_for_kwic) を使用
        # マッチ候補と「次の単語」を考慮するため、ループ範囲を調整
        for i in range(len(doc_for_kwic) - num_target_tokens): 
            candidate_doc_tokens_obj = [doc_for_kwic[j] for j in range(i, i + num_target_tokens)]
            candidate_texts_processed_lower = [
                token_text_lower for tok_obj in candidate_doc_tokens_obj
                if (token_text_lower := remove_punctuation_from_token(tok_obj.text).lower())
            ]
            
            if len(candidate_texts_processed_lower) == num_target_tokens and candidate_texts_processed_lower == target_tokens_processed_lower:
                # マッチ成功。次の単語を取得・処理
                following_word_processed = "" # 次の単語がない場合のデフォルト値
                following_token_index_in_doc = i + num_target_tokens
                if following_token_index_in_doc < len(doc_for_kwic): # ドキュメントの範囲内か確認
                    following_word_raw_text = doc_for_kwic[following_token_index_in_doc].text
                    # 次の単語も句読点除去と小文字化
                    processed_fw = remove_punctuation_from_token(following_word_raw_text).lower()
                    if processed_fw: # 空文字列でなければ採用
                        following_word_processed = processed_fw
                
                temp_token_matches.append({
                    "original_text_index": i, # マッチ開始位置の元インデックス
                    "num_tokens_in_match": num_target_tokens,
                    "following_word_processed": following_word_processed, # 処理済みの次の単語
                    # "matched_sequence_text": " ".join(candidate_texts_processed_lower) # デバッグ用
                })
        
        if temp_token_matches:
            logging.info(f"Token search initially found {len(temp_token_matches)} matches.")
            
            # 1. 全てのマッチから「次の単語（処理済み）」のリストを作成し、その出現頻度を計算
            #    次の単語が存在しない場合("")は頻度計算から除外するか、特別な値として扱う
            all_following_words = [
                item['following_word_processed'] for item in temp_token_matches if item['following_word_processed'] # 空でないもののみ
            ]
            following_word_frequencies = Counter(all_following_words)
            logging.info(f"--- [TOKEN SEARCH] Frequencies of (non-empty) following words (Top 10): {following_word_frequencies.most_common(10)} ---")

            # 2. 各マッチアイテムに「次の単語の頻度」情報を付加
            for item in temp_token_matches:
                # 次の単語が空文字列の場合は頻度0とする
                item['following_word_freq'] = following_word_frequencies.get(item['following_word_processed'], 0) if item['following_word_processed'] else 0

            # 3. ソート実行:
            #    キー1: 次の単語の頻度 (降順)
            #    キー2: 次の単語の文字列自体 (昇順 - 同じ頻度の場合の順序安定化)
            #    キー3: 元のテキストでの出現位置 (昇順 - 最終的な順序安定化)
            logging.info("--- [TOKEN SEARCH] Sorting results by 'following_word_freq' (desc), 'following_word_processed' (asc), then 'original_text_index' (asc)... ---")
            temp_token_matches.sort(key=lambda x: (
                -x['following_word_freq'],        # 頻度で降順ソート
                x['following_word_processed'],    # 次の単語の文字列で昇順ソート
                x['original_text_index']          # 元の出現位置で昇順ソート
            ))

            # 4. results_with_metadata (最終結果リスト) にコンテキスト情報を追加して格納
            for match_item in temp_token_matches:
                i = match_item['original_text_index']
                num_tokens_in_match = match_item['num_tokens_in_match']
                
                start_context_idx = max(0, i - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), i + num_tokens_in_match + backend_context_window_size)
                
                context_words_list = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                matched_start_in_context = i - start_context_idx
                
                results_with_metadata.append({
                    "context_words": context_words_list,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_start_in_context + num_tokens_in_match,
                    # デバッグ用にソートに使った情報を残しても良い (フロントエンドには不要なら削除)
                    # "debug_following_word": match_item['following_word_processed'],
                    # "debug_following_word_freq": match_item['following_word_freq'],
                    # "original_text_index": i
                })

    elif search_type == 'pos':
        target_pos_tag_query = query_input.strip().upper()
        if not target_pos_tag_query or " " in target_pos_tag_query:
            return jsonify({"error": "For POS search, enter a single valid tag."}), 400

        logging.info(f"--- [POS SEARCH] Collecting matches for POS tag '{target_pos_tag_query}' ---")
        
        temp_pos_matches = []
        for i, token in enumerate(doc_for_kwic): # spaCyのDocオブジェクトをイテレート
            if token.tag_ == target_pos_tag_query: # spaCyの細かいPOSタグを使用
                # マッチした単語のテキスト（句読点除去・小文字化）をキーとして使用
                matched_word_text_key = remove_punctuation_from_token(token.text).lower()
                if not matched_word_text_key: # 句読点除去後に空になった場合はスキップ
                    continue
                
                temp_pos_matches.append({
                    "original_text_index": i,
                    "matched_word_text_key": matched_word_text_key, # ソートや頻度計算用のキー
                    "matched_token_obj": token # 元のspaCyトークンオブジェクト (コンテキスト作成用)
                })
        
        if temp_pos_matches:
            # 1. マッチした単語（キー）の「全体での」出現頻度を計算
            matched_word_frequencies = Counter(item['matched_word_text_key'] for item in temp_pos_matches)
            logging.info(f"--- [POS SEARCH] Overall frequencies of matched POS words: {matched_word_frequencies.most_common()} ---")

            # 2. 各マッチに全体頻度と最初の出現位置を付加
            first_occurrence_map_pos = {}
            for item in temp_pos_matches:
                key = item['matched_word_text_key']
                item['matched_word_overall_freq'] = matched_word_frequencies[key]
                if key not in first_occurrence_map_pos:
                    first_occurrence_map_pos[key] = item['original_text_index']
                item['matched_word_first_occurrence'] = first_occurrence_map_pos[key]

            # 3. ソート
            temp_pos_matches.sort(key=lambda x: (
                -x['matched_word_overall_freq'],        # 全体頻度 (降順)
                x['matched_word_first_occurrence'],     # 最初の出現位置 (昇順)
                x['original_text_index']                # 実際の出現位置 (昇順)
            ))
            
            # 4. results_with_metadata にコンテキスト情報を追加して格納
            for match_item in temp_pos_matches:
                i = match_item['original_text_index']
                start_context_idx = max(0, i - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), i + 1 + backend_context_window_size)
                context_words_list = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                matched_start_in_context = i - start_context_idx
                
                results_with_metadata.append({
                    "context_words": context_words_list,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_start_in_context + 1,
                })
        # --- POS検索のロジックここまで ---

    elif search_type == 'entity':
        target_entity_type_query = query_input.strip().upper()
        if not target_entity_type_query or " " in target_entity_type_query:
            return jsonify({"error": "For Entity search, enter a single valid entity type."}), 400

        logging.info(f"--- [ENTITY SEARCH] Collecting matches for entity type '{target_entity_type_query}' ---")
        
        temp_entity_matches = []
        for ent in doc_for_kwic.ents: # spaCyのDocオブジェクトからエンティティを取得
            if ent.label_.upper() == target_entity_type_query:
                # マッチしたエンティティのテキスト（句読点除去・小文字化）をキーとして使用
                matched_entity_text_key = remove_punctuation_from_token(ent.text).lower()
                if not matched_entity_text_key: # 句読点除去後に空になった場合はスキップ
                    continue

                temp_entity_matches.append({
                    "original_text_index": ent.start, # エンティティの開始トークンインデックス
                    "matched_entity_text_key": matched_entity_text_key,
                    "num_tokens_in_match": len(ent), # エンティティを構成するトークン数
                    "entity_object": ent # 元のspaCyエンティティオブジェクト (コンテキスト作成用)
                })

        if temp_entity_matches:
            # 1. マッチしたエンティティ（キー）の「全体での」出現頻度を計算
            matched_entity_frequencies = Counter(item['matched_entity_text_key'] for item in temp_entity_matches)
            logging.info(f"--- [ENTITY SEARCH] Overall frequencies of matched entities: {matched_entity_frequencies.most_common(10)} ---")

            # 2. 各マッチに全体頻度と最初の出現位置を付加
            first_occurrence_map_entity = {}
            for item in temp_entity_matches:
                key = item['matched_entity_text_key']
                item['matched_entity_overall_freq'] = matched_entity_frequencies[key]
                if key not in first_occurrence_map_entity:
                    first_occurrence_map_entity[key] = item['original_text_index']
                item['matched_entity_first_occurrence'] = first_occurrence_map_entity[key]
            
            # 3. ソート
            temp_entity_matches.sort(key=lambda x: (
                -x['matched_entity_overall_freq'],      # 全体頻度 (降順)
                x['matched_entity_first_occurrence'],   # 最初の出現位置 (昇順)
                x['original_text_index']                # 実際の出現位置 (昇順)
            ))

            # 4. results_with_metadata にコンテキスト情報を追加して格納
            for match_item in temp_entity_matches:
                ent_start_token_idx = match_item['original_text_index']
                num_tokens = match_item['num_tokens_in_match']
                
                start_context_idx = max(0, ent_start_token_idx - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), ent_start_token_idx + num_tokens + backend_context_window_size)
                context_words_list = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                matched_start_in_context = ent_start_token_idx - start_context_idx
                
                results_with_metadata.append({
                    "context_words": context_words_list,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_start_in_context + num_tokens,
                })
        # --- Entity検索のロジックここまで ---
    
    # --- 共通の処理 (結果がない場合など) ---
    if not results_with_metadata:
        logging.info(f"Query '{query_input}' (Type: {search_type}) not found or resulted in no matches after processing.")
        return jsonify({"results": [], "error": f"The query '{query_input}' (Type: {search_type}) was not found."}), 200

    # sort_method_from_request の値に応じて最終的なソートを行う (現在はほぼ上記の頻度ソートが適用される)
    # もし 'sequential' が指定されたら、ここで出現順に再ソートするロジックが必要になるが、
    # 今回の要望は頻度優先なので、上記のソートをデフォルトとする。
    # ユーザー提供コードでは、この部分でさらにソートを上書きしていた。
    # 今回は、上記の各検索タイプ内で行ったソートを最終結果とする。
    
    final_results = results_with_metadata # 名前を変更

    # 以前のユーザーコードにあった sort_method による追加のソートは、
    # 今回の「頻度優先、その中で出現順」という明確な要件に基づき、上記で直接実装したため、
    # ここでの追加ソートは一旦コメントアウトまたは削除します。
    # もし 'sequential' や他のソート方法も維持したい場合は、ここで条件分岐が必要です。
    # logging.info(f"--- Sorting '{search_type}' results by original text index as per sort_method='{sort_method_from_request}'. ---")
    # if sort_method_from_request == 'sequential':
    #    # results_with_metadata に original_text_index があればソート可能
    #    # ただし、上記のソートで既に考慮されているため、単純な再ソートは意図と異なる可能性
    #    pass # ここで sequential ソートが必要なら実装


    logging.info(f"KWIC search successful. Returning {len(final_results)} results for type '{search_type}' (sorted by frequency then original index).")
    return jsonify({"results": final_results})


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