# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import shutil
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
import ssl
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys # 標準エラー出力もログに出力するために使用

# ロギング設定
# 本番環境では、ログをファイルや外部サービスに出力するように変更してください。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout # 標準出力にログを出す
)
# Flask自身のログレベルを設定することもできます
# logging.getLogger('flask_cors').setLevel(logging.INFO)


# SSL証明書検証の無効化は削除しました（安全なデフォルト動作）
# ssl._create_default_https_context = ssl._create_unverified_context

# NLTKデータのダウンロード確認・実行
# 本番環境では、デプロイ前にこのデータがコンテナやサーバーイメージに含まれるようにしてください。
try:
    nltk.data.find('tokenizers/punkt')
    logging.info("NLTK 'punkt' tokenizer found.")
except (nltk.downloader.DownloadError, LookupError):
     logging.warning("NLTK 'punkt' tokenizer not found. Attempting to download...")
     try:
        nltk.download('punkt')
        logging.info("NLTK 'punkt' tokenizer downloaded successfully.")
     except Exception as e:
        logging.error(f"Failed to download NLTK 'punkt': {e}")
        # ダウンロード失敗は致命的ではないが、トークン化はできなくなる

# Flaskアプリケーションインスタンス
# WSGIサーバー (Gunicornなど) はこの 'app' オブジェクトをエントリポイントとして使用します。
app = Flask(__name__)

# CORS設定
# 開発環境ではReactアプリからのアクセス(localhost:3000)を許可
# 本番環境ではReactアプリの実際のドメインを指定する必要があります（セキュリティ上必須）
allowed_origins = [
    "http://localhost:3000", # ローカル開発用
    # "https://your-production-react-app.com", # ★本番環境のドメイン例に置き換える★
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})


# ★テキストキャッシュ関連のグローバル変数とロジックは削除★
# cached_words = {}
# current_loaded_url = None


# Fetch text from URL (BeautifulSoupを使用) - セキュリティとエラーハンドリングを改善
# DoS対策としてタイムアウトを追加、HTML判定を強化
def get_text_from_url(url):
    logging.info(f"Attempting to fetch URL: {url}")
    try:
        req = urllib.request.Request(
            url,
            data=None,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' # 汎用的なUser-Agent
            }
        )
        # ★タイムアウトを設定 (秒)★ - ページの読み込みが遅すぎる場合の対策
        with urllib.request.urlopen(req, timeout=15) as response: # タイムアウトを調整可能
            # Content-Typeを確認してHTMLか判断
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()): # 小文字化して比較
                 logging.warning(f"URL is not an HTML page: {url} (Content-Type: {content_type})")
                 # クライアントに返す安全なメッセージ
                 raise ValueError(f"The provided URL does not appear to be an HTML page. (Content-Type: {content_type})")

            html = response.read()

        # BeautifulSoupでの解析
        soup = BeautifulSoup(html, 'html.parser')

        # 不要なタグを削除 (より網羅的に)
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                         'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                         'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()

        # テキスト抽出 - 単語間をスペースで区切る
        text = soup.get_text(separator=' ')
        # 連続する改行やスペースを一つにまとめる
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched and parsed URL: {url}")
        return text

    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching {url}: {e.reason}")
        # クライアントに返す一般的なエラーメッセージ
        raise ValueError(f"Could not access the provided URL. Please check the URL. Reason: {e.reason}")
    except TimeoutError: # urllib.request.urlopen の timeout が発生した場合
         logging.error(f"Timeout fetching {url} after 15 seconds.")
         raise ValueError(f"URL fetching timed out. The page took too long to respond.")
    except ValueError as e:
         # HTML判定エラーなど、ValueErrorとして発生させたもの
         logging.error(f"Value Error processing {url}: {e}")
         raise ValueError(f"{e}") # 安全なメッセージを想定（HTML判定エラーメッセージなど）
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_text_from_url for {url}: {e}\n{traceback.format_exc()}") # ★詳細をサーバーログに出力★
        # クライアントに返す一般的なエラーメッセージ
        raise ValueError(f"An unexpected error occurred while fetching or processing the URL.")


# Clean text (主に参照番号の削除など、get_text後の追加処理)
def clean_text(text):
    # [数字] のような参照番号を削除
    text = re.sub(r'\[\d+\]', '', text) # \d+で数字1個以上をより正確に
    # 連続する空白を一つにする（get_text後にも残る可能性考慮、前処理でほぼ不要だが念の為）
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route('/api/search', methods=['POST'])
def search():
    # ★キャッシュ関連のグローバル変数の宣言を削除★
    # global current_loaded_url
    # global cached_words

    data = request.json
    url = data.get('url', '').strip()
    target_input = data.get('phrase', '').strip()

    logging.info(f"Received search request for URL: {url}, Phrase: {target_input}")

    # --- 入力値の検証 (サーバー側でも重要) ---
    if not url:
        logging.warning("Search request missing URL.")
        return jsonify({"error": "Please provide a Wikipedia URL."}), 400

    # ★より堅牢なURL検証★
    try:
        parsed_url = urlparse(url)
        # スキームがhttpまたはhttpsであるか
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme received: {url}")
            return jsonify({"error": "Invalid URL scheme. Only http or https are allowed."}), 400
        # ホスト名が存在するか
        if not parsed_url.hostname:
             logging.warning(f"Invalid URL hostname received: {url}")
             return jsonify({"error": "Invalid URL hostname."}), 400

        # ★Wikipedia.org のドメインのみ許可★
        # ドメインの末尾が '.wikipedia.org' であることを確認 (サブドメインを含む)
        # さらに厳密にするなら、その前に '.' があるかもチェック: hostname == 'wikipedia.org' or hostname.lower().endswith('.wikipedia.org')
        hostname_lower = parsed_url.hostname.lower()
        if hostname_lower != 'wikipedia.org' and not hostname_lower.endswith('.wikipedia.org'):
            logging.warning(f"URL not from wikipedia.org: {url}")
            return jsonify({"error": "Only URLs from wikipedia.org are allowed."}), 400

        # ★注意: ここに内部IPアドレスへのアクセスを禁止するチェックなどを追加するとSSRF対策がより強固になりますが、複雑です。★
        # 例: resolved_ip = socket.gethostbyname(parsed_url.hostname); if is_internal_ip(resolved_ip): return error
        # ★注意: リダイレクト先のURLを検証する処理も必要になる場合がありますが、これも複雑です。★

    except ValueError: # urlparseが失敗した場合など
         logging.warning(f"Invalid URL format received: {url}")
         return jsonify({"error": "Invalid URL format."}), 400
    except Exception as e: # urlparse以外の予期せぬエラー
         logging.error(f"An unexpected error during URL parsing: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An unexpected error occurred during URL validation."}), 500

    # ★Phrase入力検証 (サーバー側でも重要)★
    if not target_input:
        logging.warning("Search request missing phrase.")
        return jsonify({"error": "Please provide a phrase to search."}), 400

    target_words = target_input.split()
    if len(target_words) > 2:
        logging.warning(f"Search phrase too long: {target_input}")
        return jsonify({"error": "Please enter one or two words only."}), 400


    # --- URLのテキスト処理 (キャッシュなし - 毎回実行) ---
    words = [] # 検索対象の単語リスト
    try:
        raw_text = get_text_from_url(url) # テキスト取得・解析（内部でエラー発生の可能性）
        text = clean_text(raw_text) # テキストクリーニング
        words = []
        if text: # クリーニング後もテキストが空でないかチェック
             # NLTKデータがダウンロードされていない場合はここでエラーになる可能性
            words = word_tokenize(text) # トークン化

        # ★キャッシュへの保存・キャッシュからの取得ロジックは削除★

    except ValueError as e:
        # get_text_from_url内で発生した安全なValueErrorを捕捉 (URLアクセス失敗、HTML判定失敗、タイムアウトなど)
        logging.warning(f"Error during URL processing for {url}: {e}")
        return jsonify({"error": f"{e}"}), 400 # get_text_from_urlからのメッセージをそのまま返す
    except Exception as e: # その他予期せぬエラー (トークン化失敗など)
         logging.error(f"An unexpected server error occurred during URL processing for {url}: {e}\n{traceback.format_exc()}")
         return jsonify({"error": "An unexpected server error occurred during URL processing."}), 500 # サーバーエラーとして返す


    # --- 検索処理 ---
    if not words:
         # URLからテキストが取得できたが、単語リストが空だった場合（例: 画像ページなど）
         # またはNLTKデータが利用できずにwordsが空の場合など
         logging.info(f"No searchable text available for URL: {url}")
         return jsonify({
             "results": [], # 結果は空リスト
             "error": f"Could not extract searchable text from the provided URL."
         }), 200 # テキストは取得できたが内容はなかった、として200を返す


    results = []
    target_input_lower = target_input.lower()
    words_lower = [word.lower() for word in words] # 検索用に単語リストを小文字化

    search_window_size = 5 # 前後の単語数

    # 1単語検索
    if len(target_words) == 1:
        target_word_lower = target_words[0].lower()
        for i, word_lower in enumerate(words_lower):
            if word_lower == target_word_lower:
                # コンテキスト全体（単語リスト）と、その中で一致した単語の開始・終了インデックスを返す
                before = words[max(0, i - search_window_size): i]
                after = words[i + 1: i + 1 + search_window_size]
                context_words = before + [words[i]] + after # コンテキスト単語リスト
                matched_start_index = len(before) # コンテキスト単語リスト内での開始位置
                matched_end_index = matched_start_index + 1 # コンテキスト単語リスト内での終了位置 + 1
                results.append({
                    "context_words": context_words,
                    "matched_start": matched_start_index,
                    "matched_end": matched_end_index
                })

    # 2単語検索
    elif len(target_words) == 2:
        target_word1_lower = target_words[0].lower()
        target_word2_lower = target_words[1].lower()
        for i in range(len(words_lower) - 1): # 2単語見るのでwords_lowerの長さ-1まで
             if words_lower[i] == target_word1_lower and words_lower[i+1] == target_word2_lower:
                before = words[max(0, i - search_window_size): i]
                after = words[i + 2: i + 2 + search_window_size]
                context_words = before + [words[i], words[i+1]] + after # コンテキスト単語リスト
                matched_start_index = len(before)
                matched_end_index = matched_start_index + 2 # 2単語なので開始+2
                results.append({
                    "context_words": context_words,
                    "matched_start": matched_start_index,
                    "matched_end": matched_end_index
                })

    # 検索結果が見つからなかった場合 (200 OK + エラーメッセージ)
    if not results:
        logging.info(f"Phrase '{target_input}' not found in text from {url}")
        return jsonify({
            "results": [], # 結果は空リスト
            "error": f"The phrase '{target_input}' was not found in the text from the provided URL." # 見つからなかったことを示すメッセージ
        }), 200 # ステータスコードは200

    # 検索結果が見つかった場合 (200 OK + 結果リスト)
    logging.info(f"Found {len(results)} occurrences for phrase '{target_input}' in text from {url}")
    return jsonify({"results": results}) # jsonifyのデフォルトが200なので指定不要


# ★開発用エントリーポイント★
# 本番環境では、GunicornなどのWSGIサーバーがこのファイルを読み込み、'app' オブジェクトを使用します。
# この if ブロックの中身は本番環境では実行されません。
if __name__ == "__main__":
    print("Flask development server starting...")
    # debug=True は開発用です。本番では必ず False にしてください。
    # host='0.0.0.0' は全てのインターフェースで待ち受けます。本番では特定のIPに制限することも検討してください。
    # threaded=True や processes=n を指定するとマルチスレッド/マルチプロセス動作を試せますが、
    # 開発用サーバーの不安定性を増す可能性があるため、基本はデフォルト（シングルスレッド）が良いです。
    app.run(debug=True, port=5000, host='0.0.0.0')