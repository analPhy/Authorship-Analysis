# app.py

# --- Imports ---
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('maxent_ne_chunker_tab') # ← この行は削除またはコメントアウトを推奨 (REQUIRED_NLTK_RESOURCESで管理)
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
import string # 句読点処理のために追加
from collections import Counter # トークン検索のソート用に前回追加 (ユーザーが追加したソートとは別)

# ... (他のML系インポートは変更なし)
from sklearn.feature_extraction.text import TfidfVectorizer
# ... (以下同様)
from langdetect import detect as lang_detect
import fugashi
from stopwordsiso import stopwords


# --- Logging Setup --- (変更なし)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

# --- NLTK Resource Check --- (変更なし、ただしpunkt_tabがユーザーコードにあったので残す)
REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab", # ユーザーコードにあったので残すが、通常punktで十分
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words"
}
# ... (NLTKリソースチェックのループは変更なし) ...

# --- 句読点除去のためのヘルパー関数と定数 ---
PUNCTUATION_SET = set(string.punctuation + '。、「」』『【】・（）　')

def remove_punctuation_from_token(token: str) -> str:
    return ''.join(char for char in token if char not in PUNCTUATION_SET)

def preprocess_tokens(tokens_list: list[str]) -> list[str]:
    """Removes punctuation from each token and filters out any empty tokens."""
    processed_tokens = [remove_punctuation_from_token(token) for token in tokens_list]
    return [token for token in processed_tokens if token]
# --- ここまで句読点除去ヘルパー ---

# ... (Fugashi, Stopwords, _SENT_RE の設定は変更なし) ...

# --- Flask App Setup --- (CORS設定はユーザーの最新版を尊重)
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


# ... (get_text_from_url_for_kwic, clean_text_for_kwic は変更なし) ...
# ... (fetch_wikipedia_text_for_authorship, mixed_sentence_tokenize_for_authorship (古いバージョン), build_sentence_dataset_for_authorship は変更なし) ...

# --- tokenize_mixed_for_authorship 関数の修正 (句読点除去と小文字化を適用) ---
def tokenize_mixed_for_authorship(text: str) -> list[str]:
    # ユーザー提供のコードは言語判定とFugashiを含んでいたので、それをベースに修正
    # ただし、以前のやり取りでタイムアウト問題のため英語tokenizeに一時的に単純化していた。
    # ここでは、ユーザー提供の多言語対応版に戻しつつ、句読点処理を追加する。
    
    initial_tokens = []
    # --- ユーザー提供の言語判定とトークン化ロジック ---
    if not _TAGGER: # Fugashiが利用できない場合
        logging.warning("Fugashi Tagger not available, defaulting to English tokenization for mixed content (authorship).")
        try:
            initial_tokens = word_tokenize(text)
        except Exception as e_tok:
            logging.error(f"NLTK word_tokenize fallback failed: {e_tok}")
            initial_tokens = text.split()
    else: # Fugashiが利用可能な場合
        try:
            lang = lang_detect(text)
        except Exception:
            logging.warning(f"Langdetect failed for text sample: '{text[:100]}'. Defaulting to English tokenization.")
            lang = "en"

        if lang == "ja":
            initial_tokens = [tok.surface for tok in _TAGGER(text)]
        else: # 他の言語 (主に英語を想定)
            try:
                initial_tokens = word_tokenize(text)
            except Exception as e_tok_en:
                logging.error(f"NLTK word_tokenize for non-Japanese text failed: {e_tok_en}")
                initial_tokens = text.split()
    # --- ここまでユーザー提供のトークン化ロジック ---

    # 句読点除去と小文字化を適用
    punctuation_removed_tokens = preprocess_tokens(initial_tokens)
    lowercased_tokens = [token.lower() for token in punctuation_removed_tokens]
    
    # logging.debug(f"Authorship tokens (processed): {lowercased_tokens[:20]}")
    return lowercased_tokens
# --- ここまで tokenize_mixed_for_authorship 関数の修正 ---


# === API Endpoints ===

@app.route('/api/search', methods=['POST'])
def kwic_search():
    data = request.json
    url = data.get('url', '').strip()
    query_input = data.get('query', '').strip() # 生の検索クエリ
    search_type = data.get('type', 'token').strip().lower()

    logging.info(f"Received KWIC search request. URL: {url}, Raw Query: '{query_input}', Type: {search_type}")

    # --- URLと基本的なクエリのバリデーション (ユーザーコードをベースに) ---
    if not url:
        logging.warning("KWIC search request missing URL.")
        return jsonify({"error": "Please provide a Web Page URL."}), 400 # メッセージを汎用化
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme: {url}")
            return jsonify({"error": "Invalid URL scheme. Only http or https are allowed."}), 400
        if not parsed_url.hostname:
             logging.warning(f"Invalid URL hostname: {url}")
             return jsonify({"error": "Invalid URL hostname."}), 400
        # ここに汎用URL対応のための is_restricted_hostname(parsed_url.hostname) チェックを追加可能 (今回は省略)
    except ValueError: # urlparseがエラーを出すことは稀だが念のため
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
    try:
        raw_text = get_text_from_url_for_kwic(url) # ユーザーの既存関数を使用
        text_cleaned = clean_text_for_kwic(raw_text)
        
        if not text_cleaned:
            logging.info(f"No text content extracted from URL: {url}")
            return jsonify({"results": [], "error": "Could not extract text content from the URL."}), 200

        initial_tokens_from_page = word_tokenize(text_cleaned)
        words_from_page_processed = preprocess_tokens(initial_tokens_from_page) # 句読点除去済み

        if not words_from_page_processed:
            logging.info(f"No searchable tokens after processing for URL: {url}")
            return jsonify({"results": [], "error": "No searchable words found in the URL after processing."}), 200

    except ValueError as e:
        logging.warning(f"Error during URL processing for KWIC search {url}: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected server error during URL/text processing: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected server error occurred during URL processing."}), 500
    # --- ここまでで words_from_page_processed が準備完了 ---

    results = []
    backend_context_window_size = 10

    if search_type == 'token':
        raw_target_tokens = query_input.split()
        target_tokens_processed = preprocess_tokens(raw_target_tokens)
        
        if not target_tokens_processed:
            logging.warning(f"Token query '{query_input}' became empty after punctuation removal.")
            return jsonify({"error": "Query became empty. Please provide a query with more than just punctuation."}), 400
        
        if not 1 <= len(target_tokens_processed) <= 5:
            logging.warning(f"Invalid token query length after punctuation removal: {target_tokens_processed} (Original: '{query_input}')")
            return jsonify({"error": "For token search, please enter one to five words (after punctuation removal)."}), 400

        words_from_page_lower = [w.lower() for w in words_from_page_processed]
        target_token_list_lower = [word.lower() for word in target_tokens_processed]
        num_target_tokens = len(target_token_list_lower)

        for i in range(len(words_from_page_lower) - num_target_tokens + 1):
            if words_from_page_lower[i : i + num_target_tokens] == target_token_list_lower:
                before = words_from_page_processed[max(0, i - backend_context_window_size): i]
                matched_segment = words_from_page_processed[i : i + num_target_tokens]
                after = words_from_page_processed[i + num_target_tokens : i + num_target_tokens + backend_context_window_size]
                context_words_list = before + matched_segment + after
                results.append({
                    "context_words": context_words_list,
                    "matched_start": len(before),
                    "matched_end": len(before) + num_target_tokens
                })

    elif search_type == 'pos':
        target_pos_tag_query = query_input.strip().upper()
        if not target_pos_tag_query or " " in target_pos_tag_query:
             return jsonify({"error": f"For POS search, please enter a single valid tag (no spaces)."}), 400
        
        logging.info(f"Attempting POS tagging for query '{target_pos_tag_query}' with {len(words_from_page_processed)} tokens.")
        try:
            # NLTKの標準的なpos_tagを使用。入力は句読点除去済みトークン。
            tagged_words = nltk.pos_tag(words_from_page_processed)
            logging.info(f"POS tagging successful. Sample: {tagged_words[:5]}")
        except Exception as e_pos_tag:
            logging.error(f"NLTK pos_tag FAILED: {e_pos_tag}\n{traceback.format_exc()}")
            return jsonify({"error": "Part-of-speech tagging failed. Check server logs."}), 500

        for i, (word, tag) in enumerate(tagged_words):
            if tag == target_pos_tag_query:
                before = words_from_page_processed[max(0, i - backend_context_window_size): i]
                matched_word = [words_from_page_processed[i]]
                after = words_from_page_processed[i + 1 : i + 1 + backend_context_window_size]
                context_words_list = before + matched_word + after
                results.append({
                    "context_words": context_words_list,
                    "matched_start": len(before), "matched_end": len(before) + 1
                })

    elif search_type == 'entity':
        target_entity_type_query = query_input.strip().upper()
        if not target_entity_type_query or " " in target_entity_type_query:
             return jsonify({"error": f"For Entity search, please enter a single valid entity type (no spaces)."}), 400

        logging.info(f"Attempting Entity Recognition for query '{target_entity_type_query}' with {len(words_from_page_processed)} tokens.")
        try:
            # NERの前処理としてPOSタギング。入力は句読点除去済みトークン。
            tagged_words_for_ner = nltk.pos_tag(words_from_page_processed)
            chunked_entities_tree = nltk.ne_chunk(tagged_words_for_ner) # Requires maxent_ne_chunker and words
            iob_tags = nltk.chunk.util.tree2conlltags(chunked_entities_tree)
            logging.info(f"Entity chunking successful. Sample IOB: {iob_tags[:5]}")
        except Exception as e_ner:
            logging.error(f"NLTK entity recognition FAILED: {e_ner}\n{traceback.format_exc()}")
            return jsonify({"error": "Entity recognition failed. Check server logs."}), 500

        idx = 0
        while idx < len(iob_tags):
            word_from_iob, _, iob_label = iob_tags[idx] 
            if iob_label.startswith('B-') and iob_label[2:] == target_entity_type_query:
                current_entity_words = [word_from_iob] 
                entity_start_idx = idx
                next_idx = idx + 1
                while next_idx < len(iob_tags) and iob_tags[next_idx][2].startswith('I-') and iob_tags[next_idx][2][2:] == target_entity_type_query:
                    current_entity_words.append(iob_tags[next_idx][0]); next_idx += 1
                else: pass # break from inner while
                
                before = words_from_page_processed[max(0, entity_start_idx - backend_context_window_size): entity_start_idx]
                after = words_from_page_processed[next_idx : next_idx + backend_context_window_size]
                context_words_list = before + current_entity_words + after
                results.append({
                    "context_words": context_words_list,
                    "matched_start": len(before), "matched_end": len(before) + len(current_entity_words)
                })
                idx = next_idx
            else: idx += 1
    else:
        logging.warning(f"Unknown search_type encountered: {search_type}")
        return jsonify({"error": "Invalid search type specified."}), 400

    # --- ユーザーが追加したソートロジック (全ての検索タイプの結果に適用) ---
    if results: # 結果がある場合のみソートを実行
        logging.info(f"Applying custom sorting to {len(results)} results for query '{query_input}' (Type: {search_type}).")
        pattern_groups = {}
        for result_item in results: # 変数名を result から result_item に変更
            # matched_end は context_words_list 内のインデックス
            # context_words_list は before + matched_segment + after で構成
            # result_item["matched_end"] は、context_words_list 内でのマッチ終了位置を示す
            
            # 正しくは、result_item["context_words"] と result_item["matched_end"] を使う
            kwic_context_words = result_item["context_words"]
            kwic_matched_end_idx = result_item["matched_end"]

            if kwic_matched_end_idx < len(kwic_context_words):
                next_words_segment = kwic_context_words[kwic_matched_end_idx:min(kwic_matched_end_idx + 3, len(kwic_context_words))]
                pattern = " ".join(word.lower() for word in next_words_segment) # パターンは小文字化
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(result_item)
            else: # マッチがコンテキストの末尾だった場合
                pattern = "" # 空のパターンとして扱うか、あるいは別のデフォルト値
                if pattern not in pattern_groups: # この処理が必要
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(result_item)


        freq_and_pos_groups = {}
        # pattern_groups に結果がない場合（全てのresultでマッチが末尾だったなど）の考慮
        if pattern_groups:
            for pattern, group in pattern_groups.items():
                freq = len(group)
                # 'matched_start' は context_words 内のインデックス。
                # 元のテキストでの出現順でソートしたい場合は、original_text_index のような情報が必要。
                # 現在の result_item には original_text_index がないので、
                # ここでは group 内の最初の result_item の matched_start を使う (近似的)。
                # より正確には、token検索で追加した original_text_index を全検索タイプで持つようにするべきだが、
                # 今回はユーザー提供のソートロジックをできるだけ活かす。
                # しかし、POS/Entity検索では original_text_index がないので、このソートは
                # トークン検索の結果に対してのみ意図通りに働く可能性がある。
                # ここでは、group内の最初の要素の `matched_start` と `context_words` を使う。
                first_item_in_group = group[0]
                # この first_pos は、あくまでそのKWICコンテキスト内でのマッチ開始位置。
                # 全体での出現順ではない。
                first_pos = first_item_in_group["matched_start"] 

                if freq not in freq_and_pos_groups:
                    freq_and_pos_groups[freq] = []
                freq_and_pos_groups[freq].append((first_pos, pattern, group))
        
        sorted_results_final = []
        if freq_and_pos_groups:
            for freq in sorted(freq_and_pos_groups.keys(), reverse=True):
                groups_at_freq = sorted(freq_and_pos_groups[freq], key=lambda x: x[0]) # first_pos (昇順)でソート
                for _, _, group_items in groups_at_freq:
                    sorted_results_final.extend(group_items)
            results = sorted_results_final # ソートされた結果で上書き
            logging.info(f"Sorting applied. Number of results remains: {len(results)}")
        else: # pattern_groupsが空だった（全てのアイテムがマッチ末尾だったなど）
            logging.info("No patterns to group by for sorting, returning results in original order of finding.")
            # results は変更なし
    # --- ソートロジックここまで ---


    if not results: # 全ての検索タイプで結果がなかった場合
        processed_query_display = ' '.join(target_tokens_processed) if search_type == 'token' and target_tokens_processed else query_input
        logging.info(f"Query '{query_input}' (Type: {search_type}, Processed: '{processed_query_display}') not found.")
        return jsonify({
            "results": [],
            "error": f"The query '{processed_query_display}' (Type: {search_type}) was not found."
        }), 200
        
    logging.info(f"Found {len(results)} occurrences for query '{query_input}' (Type: {search_type}).")
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