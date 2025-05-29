# app.py

# --- Imports ---
from collections import Counter
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk # 著者分析で一部使用
from nltk.tokenize import word_tokenize # 著者分析のデフォルトトークナイザとして使用
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
import string
import os # 環境変数やパス操作のため
# import subprocess # Build Command で spaCy モデルをダウンロードするため、アプリ内でのフォールバックは削除
import spacy # spaCyをインポート

# --- spaCyモデルとNLTKデータパスのグローバル設定 ---
_SPACY_NLP = None
_NLTK_DATA_DIR = os.environ.get('NLTK_DATA', '/opt/render/project/src/nltk_data_on_render') # download_nltk.py と同じパス

def initialize_nlp_resources():
    global _SPACY_NLP
    global _NLTK_DATA_DIR

    # spaCyモデルのロード
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm") # Build Commandでダウンロードしたモデル名を指定
        print("--- [APP STARTUP] spaCy model 'en_core_web_sm' loaded successfully. ---")
    except OSError as e:
        print(f"--- [APP STARTUP] CRITICAL ERROR: spaCy model 'en_core_web_sm' not found or failed to load: {e} ---")
        print("--- [APP STARTUP] Please ensure it is downloaded in the Build Command (e.g., python -m spacy download en_core_web_sm). ---")
        print("--- [APP STARTUP] Exiting application. ---")
        sys.exit(1)
    except Exception as e_spacy_unexpected:
        print(f"--- [APP STARTUP] An unexpected error occurred while loading spaCy model: {e_spacy_unexpected} ---")
        sys.exit(1)

    # NLTKデータパスの設定
    print(f"--- [APP STARTUP PRE-CHECK] Expected NLTK_DATA directory: {_NLTK_DATA_DIR} (from ENV or default) ---")
    if not os.path.isdir(_NLTK_DATA_DIR):
        print(f"--- [APP STARTUP PRE-CHECK] CRITICAL ERROR: NLTK data directory '{_NLTK_DATA_DIR}' does NOT exist. This should have been created by the build script. ---")
        sys.exit(1)
    else:
        print(f"--- [APP STARTUP PRE-CHECK] NLTK data directory '{_NLTK_DATA_DIR}' exists. ---")

    if _NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DATA_DIR)
    print(f"--- [APP STARTUP PRE-CHECK] NLTK data path configured to: {nltk.data.path} ---")

    # 著者分析で最低限必要なNLTKリソースの確認
    required_nltk_for_authorship = {
        'punkt': 'tokenizers/punkt', # word_tokenizeが依存
        'words': 'corpora/words'    # stopwordsisoが依存
    }
    all_nltk_ok_for_authorship = True
    print("--- [APP STARTUP] Checking NLTK resources for Authorship... ---")
    for res_id, res_check_path in required_nltk_for_authorship.items():
        try:
            nltk.data.find(res_check_path)
            print(f"--- [APP STARTUP]   [FOUND] NLTK resource for Authorship: {res_id} (Path: {res_check_path}) ---")
        except LookupError:
            print(f"--- [APP STARTUP]   [CRITICAL NOT FOUND] NLTK resource for Authorship: {res_id} (Path: {res_check_path}) ---")
            all_nltk_ok_for_authorship = False
    
    if not all_nltk_ok_for_authorship:
        print("--- [APP STARTUP] CRITICAL ERROR: Not all NLTK resources for authorship were found. ---")
        print("--- [APP STARTUP] This indicates an issue with the 'download_nltk_for_authorship.py' script in Build Command. ---")
        print("--- [APP STARTUP] Exiting application. ---")
        sys.exit(1)
    else:
        print("--- [APP STARTUP] All required NLTK resources for authorship are available. ---")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

# NLPリソースの初期化を実行
initialize_nlp_resources()

# --- Authorship Attribution Imports ---
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
_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# --- Flask App Setup ---
app = Flask(__name__)
FRONTEND_GITHUB_PAGES_ORIGIN = "https://analphy.github.io"
FRONTEND_DEV_ORIGIN_3000 = "http://localhost:3000"
FRONTEND_DEV_ORIGIN_8080 = "http://localhost:8080" # 以前使っていたポート
allowed_origins_list = [
    FRONTEND_GITHUB_PAGES_ORIGIN,
    FRONTEND_DEV_ORIGIN_3000,
    FRONTEND_DEV_ORIGIN_8080,
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins_list}})
print("--- [APP STARTUP] Flask application initialized and CORS configured. ---")
logging.info("Application setup complete. Ready for requests.")


# === Text processing for KWIC Search (Functions from original code, modified where needed) ===
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
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                        'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                        'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        content = soup.find(id='mw-content-text')
        if content:
            wiki_elements = ['toc', 'reference', 'reflist', 'navbox', 'metadata', 'catlinks', 
                           'mw-editsection', 'mw-references', 'mw-navigation', 'mw-footer',
                           'sistersitebox', 'noprint', 'mw-jump-to-nav', 'mw-indicator',
                           'mw-wiki-logo', 'mw-page-tools', 'printfooter', 'mw-revision']
            for element in content.find_all(['div', 'section', 'span', 'nav', 'footer']):
                if any(cls in (element.get('class', []) or []) for cls in wiki_elements):
                    element.decompose()
            text = content.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
        # text = soup.get_text(separator=' ') # この行は重複している可能性
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched and parsed URL (for KWIC search): {url}")
        return text
    except (URLError, HTTPError) as e_url:
        logging.error(f"URL Error fetching (KWIC search) {url}: {e_url.reason}")
        raise ValueError(f"Could not access the provided URL. Please check the URL. Reason: {e_url.reason}")
    except TimeoutError:
         logging.error(f"Timeout fetching (KWIC search) {url} after 15 seconds.")
         raise ValueError(f"URL fetching timed out. The page took too long to respond.")
    except ValueError as e_val:
          logging.error(f"Value Error processing (KWIC search) {url}: {e_val}")
          raise
    except Exception as e_gen:
        logging.error(f"An unexpected error occurred in get_text_from_url_for_kwic for {url}: {e_gen}\n{traceback.format_exc()}")
        raise ValueError(f"An unexpected error occurred while fetching or processing the URL.")

def clean_text_for_kwic(text):
    text = re.sub(r'Jump to.*?content', '', text)
    text = re.sub(r'From Wikipedia.*?encyclopedia', '', text)
    text = re.sub(r'Categories\s*:.*', '', text)
    text = re.sub(r'Hidden categories\s*:.*', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'Edit.*?section', '', text)
    text = re.sub(r'oldid=\d+', '', text)
    text = re.sub(r'\s*\(\s*disambiguation\s*\)\s*', '', text)
    text = re.sub(r'CS1.*?sources.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Articles.*?containing.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Use mdy dates.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Category:.*?(?=\n|$)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_punctuation_from_token(token: str) -> str:
    return ''.join(char for char in token if char not in PUNCTUATION_SET)

def preprocess_tokens_for_search(tokens_list: list[str]) -> list[str]:
    processed_tokens = [remove_punctuation_from_token(token) for token in tokens_list]
    return [token for token in processed_tokens if token]

# === Authorship Attribution Helpers (NLTKベースのものはそのまま流用) ===
def fetch_wikipedia_text_for_authorship(url: str) -> str:
    # (この関数は元のコードのままとします。エラーハンドリング等も含む)
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
    except (URLError, HTTPError) as e_url_auth: # Renamed variable
        logging.error(f"URL Error fetching (authorship) {url}: {e_url_auth.reason}")
        raise ValueError(f"Could not access the URL for authorship. Reason: {e_url_auth.reason}")
    except TimeoutError:
        logging.error(f"Timeout fetching (authorship) {url}")
        raise ValueError("URL fetching for authorship timed out.")
    except Exception as err_fetch_auth: # Renamed variable
        logging.error(f"Error fetching URL (authorship) {url}: {err_fetch_auth}\n{traceback.format_exc()}")
        raise ValueError(f"An unexpected error occurred while fetching URL for authorship.")
    
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                     'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                     'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
        tag.decompose()
    content = soup.find(id='mw-content-text')
    if content:
        for section in content.find_all(['div', 'section'], class_=['toc', 'reference', 'reflist', 'navbox', 'metadata', 'catlinks']):
            section.decompose()
        for element in content.find_all(['span', 'div'], class_=['mw-editsection', 'reference', 'reflist']):
            element.decompose()
        text = content.get_text(separator=' ')
    else:
        text = soup.get_text(separator=' ')
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'Edit.*?section', '', text)
    text = re.sub(r'Jump to.*?content', '', text)
    text = re.sub(r'From Wikipedia.*?encyclopedia', '', text)
    text = re.sub(r'Categories\s*:.*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    logging.info(f"Successfully fetched and parsed Wikipedia text for authorship from URL: {url}")
    return text

def mixed_sentence_tokenize_for_authorship(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def tokenize_mixed_for_authorship(text: str):
    # 著者分析ではNLTKのword_tokenizeを使い続ける (punktとwordsに依存)
    try:
        initial_tokens = word_tokenize(text)
    except Exception as e_tok_auth:
        logging.error(f"NLTK word_tokenize failed during authorship tokenization: {e_tok_auth}")
        initial_tokens = text.split() # フォールバック
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
    return "Hello, World! This is the home page."

@app.route('/api/search', methods=['POST'])
def kwic_search():
    data = request.json
    url = data.get('url', '').strip()
    query_input = data.get('query', '').strip()
    search_type = data.get('type', 'token').strip().lower()
    sort_method = data.get('sort_method', 'sequential')

    logging.info(f"Received KWIC search request. URL: {url}, Raw Query: '{query_input}', Type: {search_type}, Sort: {sort_method}")

    if not url: return jsonify({"error": "Please provide a Web Page URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https'] or not parsed_url.hostname:
            return jsonify({"error": "Invalid URL. Only http/https from valid hostnames are allowed."}), 400
    except Exception: return jsonify({"error": "Invalid URL format."}), 400
    if not query_input: return jsonify({"error": "Please provide a search query."}), 400

    text_from_page_for_kwic = ""
    try:
        raw_text = get_text_from_url_for_kwic(url)
        text_from_page_for_kwic = clean_text_for_kwic(raw_text)
        if not text_from_page_for_kwic:
            return jsonify({"results": [], "error": "Could not extract text content from the URL."}), 200
    except ValueError as e: return jsonify({"error": str(e)}), 400
    except Exception as e_proc:
        logging.error(f"Unexpected server error during URL/text processing for KWIC: {e_proc}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during URL processing."}), 500

    results = []
    backend_context_window_size = 10

    doc_for_kwic = None
    if _SPACY_NLP:
        try:
            doc_for_kwic = _SPACY_NLP(text_from_page_for_kwic)
        except Exception as e_spa_proc:
            logging.error(f"spaCy processing failed for KWIC text: {e_spa_proc}\n{traceback.format_exc()}")
            return jsonify({"error": "Text processing failed on the server (spaCy)."}), 500
    else:
        return jsonify({"error": "NLP model (spaCy) is not available. Cannot process search."}), 500

    # spaCyのDocオブジェクトから元のケースの単語リストを生成 (KWIC表示用)
    words_from_page_original_case_spacy = [token.text for token in doc_for_kwic]
    # 検索比較用に小文字化・句読点除去したリスト (spaCyのトークンに基づいて)
    words_from_page_processed_lower_spacy = [
        remove_punctuation_from_token(token.text).lower() 
        for token in doc_for_kwic 
        if remove_punctuation_from_token(token.text) # 空のトークンを除去
    ]


    if search_type == 'token':
        raw_target_tokens = query_input.split()
        target_tokens_processed_lower = [remove_punctuation_from_token(word).lower() for word in raw_target_tokens if remove_punctuation_from_token(word)]

        if not target_tokens_processed_lower:
            return jsonify({"error": "Query became empty after processing."}), 400
        if not 1 <= len(target_tokens_processed_lower) <= 5:
            return jsonify({"error": "For token search, please enter one to five words."}), 400

        num_target_tokens = len(target_tokens_processed_lower)
        temp_results_for_sorting = []
        all_observed_following_words = []
        
        # マッチングは words_from_page_processed_lower_spacy で行う
        # コンテキスト表示は words_from_page_original_case_spacy を使う
        # words_from_page_processed_lower_spacy のインデックスが doc_for_kwic のトークンインデックスと
        # 直接対応しない可能性があるため注意（句読点のみのトークンが除去されるため）
        # より正確には、doc_for_kwic をループし、条件に合うトークンシーケンスを探す
        
        current_processed_idx = 0
        for i in range(len(doc_for_kwic) - num_target_tokens + 1):
            # doc_for_kwic から num_target_tokens 分のトークンを取得し、
            # それらを句読点除去・小文字化して検索クエリと比較する
            candidate_tokens_original = [doc_for_kwic[j] for j in range(i, i + num_target_tokens)]
            candidate_tokens_processed_lower = [
                remove_punctuation_from_token(tok.text).lower() 
                for tok in candidate_tokens_original 
                if remove_punctuation_from_token(tok.text) # 句読点のみのトークンは無視
            ]
            
            # 処理済み候補トークン数が検索クエリのトークン数と一致するか確認
            if len(candidate_tokens_processed_lower) == num_target_tokens and candidate_tokens_processed_lower == target_tokens_processed_lower:
                # マッチした場合、コンテキストは doc_for_kwic から取得
                # マッチした最初のトークンのインデックスは i
                # マッチした最後のトークンの次のインデックスは i + num_target_tokens

                # 元のテキストでのコンテキストを構築
                # ここでの i は doc_for_kwic のトークンインデックス
                start_context_idx = max(0, i - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), i + num_target_tokens + backend_context_window_size)
                
                context_doc_tokens = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                
                matched_start_in_context = i - start_context_idx
                matched_end_in_context = matched_start_in_context + num_target_tokens
                
                current_following_word = ""
                following_token_idx_in_doc = i + num_target_tokens
                if following_token_idx_in_doc < len(doc_for_kwic):
                    current_following_word = doc_for_kwic[following_token_idx_in_doc].text.lower()
                    all_observed_following_words.append(current_following_word)

                temp_results_for_sorting.append({
                    "context_words": context_doc_tokens,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_end_in_context,
                    "original_text_index": i, # doc_for_kwic における開始トークンインデックス
                    "raw_following_word": current_following_word
                })

        if not temp_results_for_sorting:
            logging.info(f"Token query '{query_input}' (Processed: '{' '.join(target_tokens_processed_lower)}') not found.")
        else:
            overall_following_word_frequencies = Counter(all_observed_following_words)
            results_with_frequency_info = []
            for item in temp_results_for_sorting:
                item['overall_following_word_frequency'] = overall_following_word_frequencies.get(item['raw_following_word'], 0)
                results_with_frequency_info.append(item)
            results_with_frequency_info.sort(key=lambda x: (-x['overall_following_word_frequency'], x['original_text_index']))
            results = results_with_frequency_info

    elif search_type == 'pos':
        target_pos_tag_query = query_input.strip().upper()
        if not target_pos_tag_query or " " in target_pos_tag_query:
             return jsonify({"error": "For POS search, please enter a single valid tag."}), 400

        for i, token in enumerate(doc_for_kwic):
            if token.tag_ == target_pos_tag_query: # spaCyの token.tag_ (詳細な品詞タグ) を使用
                start_context_idx = max(0, i - backend_context_window_size)
                end_context_idx = min(len(doc_for_kwic), i + 1 + backend_context_window_size)
                
                context_doc_tokens = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                
                matched_start_in_context = i - start_context_idx
                matched_end_in_context = matched_start_in_context + 1
                
                results.append({
                    "context_words": context_doc_tokens,
                    "matched_start": matched_start_in_context,
                    "matched_end": matched_end_in_context
                })

    elif search_type == 'entity':
        target_entity_type_query = query_input.strip().upper()
        if not target_entity_type_query or " " in target_entity_type_query:
            return jsonify({"error": "For Entity search, please enter a single valid entity type."}), 400

        if doc_for_kwic:
            for ent in doc_for_kwic.ents:
                if ent.label_.upper() == target_entity_type_query:
                    start_token_doc_idx = ent.start
                    end_token_doc_idx = ent.end

                    start_context_idx = max(0, start_token_doc_idx - backend_context_window_size)
                    end_context_idx = min(len(doc_for_kwic), end_token_doc_idx + backend_context_window_size)

                    context_doc_tokens = [doc_for_kwic[k].text for k in range(start_context_idx, end_context_idx)]
                    
                    matched_start_in_context = start_token_doc_idx - start_context_idx
                    matched_end_in_context = matched_start_in_context + (end_token_doc_idx - start_token_doc_idx)
                    
                    results.append({
                        "context_words": context_doc_tokens,
                        "matched_start": matched_start_in_context,
                        "matched_end": matched_end_in_context
                    })
        else:
            logging.warning("spaCy NLP model was not available for entity search.")


    if not results:
        logging.info(f"Query '{query_input}' (Type: {search_type}) not found in text from {url}")
        return jsonify({
            "results": [],
            "error": f"The query '{query_input}' (Type: {search_type}) was not found."
        }), 200

    sorted_results = results # デフォルトは出現順 (token検索の場合は既にソート済みの場合あり)
    if sort_method == 'token' and search_type == 'token': # tokenソートはtoken検索の結果にのみ適用
        pass # results は既にソートされている想定
    elif sort_method == 'pos':
        pos_groups = {}
        if _SPACY_NLP:
            for result_item in results:
                context = result_item['context_words']
                match_end_idx_in_context = result_item['matched_end']
                if match_end_idx_in_context < len(context):
                    next_word_text = context[match_end_idx_in_context]
                    next_word_doc = _SPACY_NLP(next_word_text) # 次の単語だけを処理
                    if next_word_doc and len(next_word_doc) > 0:
                        pos_tag = next_word_doc[0].tag_
                        if pos_tag not in pos_groups:
                            pos_groups[pos_tag] = []
                        pos_groups[pos_tag].append(result_item)
            if pos_groups:
                temp_sorted_list = []
                pos_freqs = sorted(pos_groups.items(), key=lambda item: len(item[1]), reverse=True)
                for tag, group_items in pos_freqs:
                    temp_sorted_list.extend(group_items)
                sorted_results = temp_sorted_list
        else:
            logging.warning("spaCy model not available for POS sorting.")

    logging.info(f"Found {len(results)} raw occurrences for query '{query_input}'. Returning {len(sorted_results)} after sorting by '{sort_method}'.")
    return jsonify({"results": sorted_results})


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
        logging.info("Downloading Wikipedia pages for authorship...")
        text_a = fetch_wikipedia_text_for_authorship(url_a)
        text_b = fetch_wikipedia_text_for_authorship(url_b)
        if not text_a or not text_b:
            return jsonify({"error": "Failed to fetch content from one or both Wikipedia pages."}), 500
        
        logging.info("Splitting into sentences & labelling for authorship...")
        sentences_a, labels_a = build_sentence_dataset_for_authorship(text_a, "AuthorA")
        sentences_b, labels_b = build_sentence_dataset_for_authorship(text_b, "AuthorB")
        
        if not sentences_a or not sentences_b or len(sentences_a) < 2 or len(sentences_b) < 2:
            return jsonify({"error": "Could not extract enough valid sentences (min 30 chars, min 2 per author) for analysis."}), 400

        all_sentences = sentences_a + sentences_b
        all_labels = labels_a + labels_b
        
        if len(set(all_labels)) < 2:
             return jsonify({"error": "Could not gather sentences from both sources to perform comparison."}), 400

        unique_labels_counts = Counter(all_labels)
        if any(c < 2 for c in unique_labels_counts.values()) or len(all_sentences) < 5 :
            return jsonify({"error": "Not enough sentences from each author (min 2) or too few sentences overall (min 5) for reliable model training."}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        logging.info(f"Authorship Training samples: {len(X_train)} — Test samples: {len(X_test)}")

        if not X_train or not X_test:
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
    except Exception as e_auth:
        logging.error(f"An unexpected error occurred during authorship analysis: {e_auth}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during authorship analysis."}), 500

if __name__ == "__main__":
    # ローカル開発時のみ実行される
    print("--- Flask development server starting (app.py __main__) ---")
    # initialize_nlp_resources() は既にグローバルスコープで呼び出されている
    
    if not _SPACY_NLP:
        print("CRITICAL: spaCy model could not be loaded in __main__. Exiting.")
        sys.exit(1)
    
    # fugaishiタガーの確認 (initialize_nlp_resourcesの外なのでここで良い)
    if not _TAGGER:
        print("Warning: Fugashi Tagger failed to initialize in __main__. Japanese NLP features for authorship attribution will be limited.")
    
    print("Starting Flask app server via app.run()...")
    app.run(debug=True, port=8080, host='0.0.0.0') # ローカルテスト用にポート8080を使用