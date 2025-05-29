# app.py

# --- Imports ---
from collections import Counter
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import urllib.request
from urllib.error import URLError, HTTPError
import nltk
# word_tokenize は直接使わず、TreebankWordTokenizer を使用
from nltk.tokenize import TreebankWordTokenizer
# sent_tokenize も直接使わず、ロード済みのPunktSentenceTokenizerを使用
# from nltk.tokenize.punkt import PunktSentenceTokenizer # nltk.data.load() を使うので直接インポートは不要な場合も
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import traceback
import logging
import sys
import string

# --- Authorship Attribution Imports --- (変更なし)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  # ← ★この行を確認・追加★
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score
# ... (以下同様)
from langdetect import detect as lang_detect
import fugashi
from stopwordsiso import stopwords



# --- Punctuation Handling --- (変更なし)
PUNCTUATION_SET = set(string.punctuation + '。、「」』『【】・（）　')
def remove_punctuation_from_token(token: str) -> str:
    return ''.join(char for char in token if char not in PUNCTUATION_SET)
def preprocess_tokens(tokens_list: list[str]) -> list[str]:
    if not isinstance(tokens_list, list):
        logging.warning(f"preprocess_tokens received non-list input: {type(tokens_list)}")
        return []
    processed_tokens = [remove_punctuation_from_token(str(token)) for token in tokens_list]
    return [token for token in processed_tokens if token]

# --- Logging Setup --- (変更なし)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', stream=sys.stdout)

# --- NLTK Resource Check ---
REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",  # For PunktSentenceTokenizer (english.pickle)
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
    "words": "corpora/words"
}
# ... (NLTKリソースチェックのループは変更なし、前回提供の通り) ...
logging.info("Checking NLTK resources...")
for download_key, path_to_find in REQUIRED_NLTK_RESOURCES.items():
    try:
        nltk.data.find(path_to_find)
        logging.info(f"NLTK resource '{download_key}' (for path '{path_to_find}') found.")
    except LookupError:
        logging.warning(f"NLTK resource '{download_key}' not found. Attempting to download '{download_key}'...")
        try:
            download_successful = nltk.download(download_key, quiet=False)
            if download_successful is None or download_successful:
                logging.info(f"NLTK download command for '{download_key}' executed.")
                nltk.data.find(path_to_find)
                logging.info(f"NLTK resource '{download_key}' now found after download.")
            else: # pragma: no cover
                logging.error(f"NLTK download for '{download_key}' returned False. Verifying...")
                try: nltk.data.find(path_to_find); logging.info(f"'{download_key}' found despite False return.")
                except LookupError: logging.error(f"CRITICAL: '{download_key}' still not found.")
        except Exception as e_download: # pragma: no cover
            logging.error(f"Error during NLTK resource '{download_key}' download/verification: {e_download}")
            logging.error(f"Manual download may be needed: python -m nltk.downloader {download_key}")


# --- Global NLTK Tokenizer Instances ---
_ENG_SENT_TOKENIZER = None
_TB_WORD_TOKENIZER = TreebankWordTokenizer() 
try:
    # Load the standard English Punkt sentence tokenizer model from the 'punkt' resource
    _ENG_SENT_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
    logging.info("Standard English PunktSentenceTokenizer loaded successfully from pickle.")
except LookupError: # pragma: no cover
    logging.error("CRITICAL: NLTK 'punkt/english.pickle' not found. Sentence tokenization will fail.")
except Exception as e: # pragma: no cover
    logging.error(f"Error loading English PunktSentenceTokenizer: {e}")


# --- Global Settings for Authorship Attribution --- (変更なし)
# ... (_TAGGER, _EN_SW, _JA_SW, _ALL_SW, _SENT_RE) ...
_TAGGER = None
try: _TAGGER = fugashi.Tagger(); logging.info("fugashi Tagger initialized.")
except Exception as e: logging.error(f"Fugashi Tagger init failed: {e}")
_EN_SW, _JA_SW = set(), set()
try: _EN_SW = stopwords("en"); _JA_SW = stopwords("ja"); logging.info("Stopwords loaded.")
except Exception as e: logging.error(f"Stopwords load failed: {e}")
_ALL_SW = sorted(list(_EN_SW.union(_JA_SW)))
_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# --- Flask App Setup --- (変更なし)
# ... (CORS設定含む) ...
app = Flask(__name__)
FRONTEND_GITHUB_PAGES_ORIGIN = "https://analphy.github.io"
FRONTEND_DEV_ORIGIN_3000 = "http://localhost:3000"
FRONTEND_DEV_ORIGIN_8080 = "http://localhost:8080"
allowed_origins_list = [FRONTEND_GITHUB_PAGES_ORIGIN, FRONTEND_DEV_ORIGIN_3000, FRONTEND_DEV_ORIGIN_8080]
CORS(app, resources={r"/api/*": {"origins": allowed_origins_list}})


# === Consolidated Text processing function === (変更なし)
def get_text_from_url(url: str, purpose: str = "generic text extraction") -> str:
    # ... (前回提供の get_text_from_url 関数の実装) ...
    logging.info(f"Attempting to fetch URL (for {purpose}): {url}")
    try:
        req = urllib.request.Request(
            url, data=None,
            headers={'User-Agent': f'Mozilla/5.0 (compatible; TextAnalysisToolBot/1.0; AppPurpose/{purpose})'}
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            content_type = response.getheader('Content-Type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                 raise ValueError(f"URL content type ({content_type}) not supported.")
            html_or_text_bytes = response.read()
            try: html_or_text_content = html_or_text_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try: html_or_text_content = html_or_text_bytes.decode('latin-1')
                except UnicodeDecodeError: 
                    try: html_or_text_content = html_or_text_bytes.decode(response.headers.get_content_charset() or 'iso-8859-1', errors='replace')
                    except Exception: raise ValueError("Could not decode content.")
        if 'text/html' in content_type:
            soup = BeautifulSoup(html_or_text_content, 'html.parser')
            tags_to_remove = ['script', 'style', 'head', 'link', 'meta', 'noscript', 'nav', 'footer', 'aside', 'form', 'header', 'figure', 'figcaption', 'iframe', 'svg', 'canvas', 'button', 'input', 'select', 'textarea', 'img', 'audio', 'video', 'object', 'embed', 'sup']
            selectors_to_remove = ['#cookie-banner', '.cookie-consent', '#sidebar', '.sidebar', '.advertisement', '.ad-container', '.ad', '.ads', '.modal', '.popup', '.site-header', '.site-nav', '.site-navigation', '.site-footer', '.footer-links', '.bottom-nav', '.comments', '#comments', '.comment-section', '.related-posts', '.share-buttons']
            for tag_name in tags_to_remove:
                for tag in soup.find_all(tag_name): tag.decompose()
            for selector in selectors_to_remove:
                for tag in soup.select(selector): tag.decompose()
            main_content_selectors = ['article', 'main', '[role="main"]', '.main-content', '#main-content', '.entry-content', '.post-content', '.article-body']
            main_text_area = None
            for selector in main_content_selectors:
                main_text_area = soup.select_one(selector)
                if main_text_area: break
            if main_text_area: text = main_text_area.get_text(separator=' ', strip=True)
            else: text = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
        else: text = html_or_text_content
        text = re.sub(r'\[\d+\]', '', text); text = re.sub(r'\s+', ' ', text).strip()
        if not text: logging.warning(f"No significant text extracted from URL ({purpose}): {url}")
        logging.info(f"Successfully fetched and parsed URL ({purpose}): {url}")
        return text
    except (URLError, HTTPError) as e: raise ValueError(f"Could not access URL. Reason: {getattr(e, 'reason', 'Network/HTTP error')}")
    except TimeoutError: raise ValueError("URL fetching timed out.")
    except ValueError as e: raise
    except Exception as e:
        logging.error(f"Unexpected error in get_text_from_url ({purpose}) for {url}: {e}\n{traceback.format_exc()}")
        raise ValueError("Unexpected error fetching/processing URL.")


# --- Tokenizer for Authorship (Uses pre-loaded _ENG_SENT_TOKENIZER and _TB_WORD_TOKENIZER) ---
def tokenize_mixed_for_authorship(text: str) -> list[str]:
    initial_tokens = []
    # Use global tokenizers pre-loaded at startup
    global _ENG_SENT_TOKENIZER, _TB_WORD_TOKENIZER, _TAGGER 

    if not _TAGGER: # Fugashi not available
        logging.warning("Fugashi Tagger not available, using NLTK standard tokenization for authorship (English assumed).")
        try: 
            if _ENG_SENT_TOKENIZER:
                sentences_auth = _ENG_SENT_TOKENIZER.tokenize(text)
                for sentence in sentences_auth: initial_tokens.extend(_TB_WORD_TOKENIZER.tokenize(sentence))
            else: 
                logging.error("English sentence tokenizer not loaded for authorship. Falling back to simple split.")
                initial_tokens = text.split() 
        except Exception as e_tok:
            logging.error(f"NLTK tokenization fallback failed in authorship: {e_tok}\n{traceback.format_exc()}"); initial_tokens = text.split()
    else: # Fugashi available
        try: lang = lang_detect(text)
        except Exception: logging.warning(f"Langdetect failed for authorship sample: '{text[:100]}'. Defaulting to English."); lang = "en"
        
        if lang == "ja":
            initial_tokens = [tok.surface for tok in _TAGGER(text)]
        else: # Default to NLTK's PunktSentenceTokenizer + TreebankWordTokenizer for other languages
            try: 
                if _ENG_SENT_TOKENIZER:
                    sentences_auth = _ENG_SENT_TOKENIZER.tokenize(text)
                    for sentence in sentences_auth: initial_tokens.extend(_TB_WORD_TOKENIZER.tokenize(sentence))
                else:
                    logging.error("English sentence tokenizer not loaded for non-JA authorship. Falling back to simple split.")
                    initial_tokens = text.split()
            except Exception as e_tok_en: logging.error(f"NLTK tokenization for non-JA in authorship: {e_tok_en}\n{traceback.format_exc()}"); initial_tokens = text.split()
    
    punctuation_removed_tokens = preprocess_tokens(initial_tokens)
    lowercased_tokens = [token.lower() for token in punctuation_removed_tokens]
    return lowercased_tokens

# --- Sentence processing for Authorship (User's existing function) ---
def mixed_sentence_tokenize_for_authorship(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

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
    search_type = data.get('type', 'token').strip().lower()

    logging.info(f"Received KWIC search. URL: {url}, Raw Query: '{query_input}', Type: {search_type}")
    global _ENG_SENT_TOKENIZER, _TB_WORD_TOKENIZER # Use pre-loaded tokenizers

    if not url: return jsonify({"error": "Please provide a Web Page URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https'] or not parsed_url.hostname:
             return jsonify({"error": "Invalid URL (scheme/hostname)."}), 400
    except Exception as e: return jsonify({"error": f"Invalid URL format: {e}"}), 400
    if not query_input: return jsonify({"error": "Please provide a search query."}), 400

    try:
        text_content = get_text_from_url(url, "KWIC search")
        if not text_content: return jsonify({"results": [], "error": "Could not extract text content."}), 200
        
        initial_tokens_from_page = []
        if _ENG_SENT_TOKENIZER: # Check if tokenizer loaded successfully
            sentences = _ENG_SENT_TOKENIZER.tokenize(text_content)
            for sentence in sentences: 
                initial_tokens_from_page.extend(_TB_WORD_TOKENIZER.tokenize(sentence))
            logging.info(f"Tokenization for KWIC successful. Tokens: {len(initial_tokens_from_page)}")
        else:
            logging.error("CRITICAL: English sentence tokenizer (_ENG_SENT_TOKENIZER) is not available for KWIC search!")
            return jsonify({"error": "Server-side tokenizer initialization failed. Please contact support."}), 500
            
        words_from_page_processed = preprocess_tokens(initial_tokens_from_page)
        if not words_from_page_processed: return jsonify({"results": [], "error": "No searchable words after processing."}), 200
    except ValueError as e: return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"KWIC: Error processing URL/text: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Error processing URL content."}), 500

    results = []
    backend_context_window_size = 10
    display_query_for_not_found = query_input 

    if search_type == 'token':
        raw_target_tokens = query_input.split()
        target_tokens_processed = preprocess_tokens(raw_target_tokens)
        display_query_for_not_found = ' '.join(target_tokens_processed) if target_tokens_processed else query_input
        if not target_tokens_processed: return jsonify({"error": "Query empty after punctuation removal."}), 400
        if not 1 <= len(target_tokens_processed) <= 5: return jsonify({"error": "Token search: 1-5 words (after punctuation removal)."}), 400

        words_from_page_lower = [w.lower() for w in words_from_page_processed]
        target_token_list_lower = [word.lower() for word in target_tokens_processed]
        num_target_tokens = len(target_token_list_lower)
        for i in range(len(words_from_page_lower) - num_target_tokens + 1):
            if words_from_page_lower[i : i + num_target_tokens] == target_token_list_lower:
                before = words_from_page_processed[max(0, i - backend_context_window_size): i]
                matched_segment = words_from_page_processed[i : i + num_target_tokens]
                after = words_from_page_processed[i + num_target_tokens : i + num_target_tokens + backend_context_window_size]
                results.append({"context_words": before + matched_segment + after, "matched_start": len(before), "matched_end": len(before) + num_target_tokens, "original_text_index": i})
    
    elif search_type == 'pos':
        target_pos_tag_query = query_input.strip().upper()
        display_query_for_not_found = target_pos_tag_query
        if not target_pos_tag_query or " " in target_pos_tag_query: return jsonify({"error": "Invalid POS tag query."}), 400
        logging.info(f"POS tagging for: '{target_pos_tag_query}' on {len(words_from_page_processed)} tokens.")
        try: tagged_words = nltk.pos_tag(words_from_page_processed)
        except Exception as e_pos:
            logging.error(f"NLTK pos_tag FAILED: {e_pos}\n{traceback.format_exc()}"); return jsonify({"error": "POS tagging failed on server."}), 500
        for i, (word, tag) in enumerate(tagged_words):
            if tag == target_pos_tag_query:
                before = words_from_page_processed[max(0, i - backend_context_window_size): i]
                after = words_from_page_processed[i + 1 : i + 1 + backend_context_window_size]
                results.append({"context_words": before + [words_from_page_processed[i]] + after, "matched_start": len(before), "matched_end": len(before) + 1, "original_text_index": i})

    elif search_type == 'entity':
        target_entity_type_query = query_input.strip().upper()
        display_query_for_not_found = target_entity_type_query
        if not target_entity_type_query or " " in target_entity_type_query: return jsonify({"error": "Invalid Entity type query."}), 400
        logging.info(f"Entity recognition for: '{target_entity_type_query}' on {len(words_from_page_processed)} tokens.")
        try:
            tagged_words_for_ner = nltk.pos_tag(words_from_page_processed)
            chunked_entities_tree = nltk.ne_chunk(tagged_words_for_ner)
            iob_tags = nltk.chunk.util.tree2conlltags(chunked_entities_tree)
        except Exception as e_ner:
            logging.error(f"NLTK NER FAILED: {e_ner}\n{traceback.format_exc()}"); return jsonify({"error": "Entity recognition failed on server."}), 500
        idx = 0
        while idx < len(iob_tags):
            word_from_iob, _, iob_label = iob_tags[idx]
            if iob_label.startswith('B-') and iob_label[2:] == target_entity_type_query:
                current_entity_words = [word_from_iob]; entity_start_idx_in_iob = idx; next_idx = idx + 1
                while next_idx < len(iob_tags) and iob_tags[next_idx][2].startswith('I-') and iob_tags[next_idx][2][2:] == target_entity_type_query:
                    current_entity_words.append(iob_tags[next_idx][0]); next_idx += 1
                before = words_from_page_processed[max(0, entity_start_idx_in_iob - backend_context_window_size): entity_start_idx_in_iob]
                after = words_from_page_processed[next_idx : next_idx + backend_context_window_size]
                results.append({"context_words": before + current_entity_words + after, "matched_start": len(before), "matched_end": len(before) + len(current_entity_words), "original_text_index": entity_start_idx_in_iob})
                idx = next_idx
            else: idx += 1
    else: return jsonify({"error": "Invalid search type."}), 400

    # --- User's custom sorting logic ---
    if results:
        logging.info(f"Applying custom sorting to {len(results)} initial results.")
        pattern_groups = {}
        for r_item in results: 
            kwic_ctx_words = r_item["context_words"]; kwic_match_end = r_item["matched_end"]; pattern_key = "" 
            if kwic_match_end < len(kwic_ctx_words):
                next_words_seg = kwic_ctx_words[kwic_match_end:min(kwic_match_end + 3, len(kwic_ctx_words))]
                if next_words_seg: pattern_key = " ".join(word.lower() for word in next_words_seg)
            if pattern_key not in pattern_groups: pattern_groups[pattern_key] = []
            pattern_groups[pattern_key].append(r_item)
        freq_and_pos_groups = {}
        if pattern_groups:
            for pattern, group in pattern_groups.items():
                freq = len(group); first_pos = min(item.get("original_text_index", float('inf')) for item in group)
                if freq not in freq_and_pos_groups: freq_and_pos_groups[freq] = []
                freq_and_pos_groups[freq].append((first_pos, pattern, group))
        sorted_results_final = []
        if freq_and_pos_groups:
            for freq_key in sorted(freq_and_pos_groups.keys(), reverse=True):
                groups_at_freq = sorted(freq_and_pos_groups[freq_key], key=lambda x: x[0]) 
                for _, _, group_items_list in groups_at_freq: sorted_results_final.extend(group_items_list)
            results = sorted_results_final
            logging.info(f"Custom sorting applied. Result count: {len(results)}")
        elif results: 
            logging.info("No distinct patterns for custom sorting. Applying fallback sort by original_text_index.")
            if all("original_text_index" in r for r in results): results.sort(key=lambda x: x.get("original_text_index", 0))
            else: logging.warning("Fallback sort by original_text_index skipped: key not in all items.")
    # --- End of custom sorting logic ---

    if not results:
        logging.info(f"Query '{query_input}' (Type: {search_type}, Display Query: '{display_query_for_not_found}') ultimately yielded no results.")
        return jsonify({"results": [], "error": f"The query '{display_query_for_not_found}' (Type: {search_type}) was not found."}), 200
    logging.info(f"Found and returning {len(results)} occurrences for query '{query_input}' (Type: {search_type}).")
    return jsonify({"results": results})

# === Authorship Analysis Endpoint ===
@app.route('/api/authorship', methods=['POST'])
def authorship_analysis():
    data = request.json
    url_a, url_b = data.get('url_a', '').strip(), data.get('url_b', '').strip()
    logging.info(f"Authorship request: URL A: {url_a}, URL B: {url_b}")

    if not url_a or not url_b: return jsonify({"error": "Please provide two Web Page URLs."}), 400
    
    # User's original logic was specific to Wikipedia for authorship.
    # If general URLs are desired, this validation should be removed/changed
    # and get_text_from_url should be used instead of fetch_wikipedia_text_for_authorship.
    for i, url_val in enumerate([url_a, url_b]):
        label = "A" if i == 0 else "B"
        try:
            parsed = urlparse(url_val)
            if not parsed.scheme in ['http', 'https'] or not parsed.hostname: raise ValueError("Invalid URL")
            hostname_lower = parsed.hostname.lower()
            if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
                return jsonify({"error": f"URL for Source {label} ('{url_val}') must be from wikipedia.org (current authorship logic)."}), 400
        except ValueError: return jsonify({"error": f"Invalid URL format for Source {label}: {url_val}"}), 400

    try:
        # Using user's fetch_wikipedia_text_for_authorship for this endpoint as per their code structure
        # For general URLs, replace with get_text_from_url
        text_a = get_text_from_url(url_a, "Authorship Source A")
        text_b = get_text_from_url(url_b, "Authorship Source B")
        if not text_a or not text_b: return jsonify({"error": "Failed to fetch content for authorship."}), 500

        sentences_a, labels_a = build_sentence_dataset_for_authorship(text_a, "AuthorA")
        sentences_b, labels_b = build_sentence_dataset_for_authorship(text_b, "AuthorB")
        if not sentences_a or not sentences_b:
            error_msg = "Could not extract enough valid sentences for analysis. "
            if not sentences_a: error_msg += f"Problem with URL A. "
            if not sentences_b: error_msg += f"Problem with URL B. "
            return jsonify({"error": error_msg.strip()}), 400

        all_sentences, all_labels = sentences_a + sentences_b, labels_a + labels_b
        if len(set(all_labels)) < 2: return jsonify({"error": "Need sentences from two distinct sources."}), 400
        
        MIN_SAMPLES = 2 
        if all_labels.count("AuthorA") < MIN_SAMPLES or all_labels.count("AuthorB") < MIN_SAMPLES or len(all_sentences) < 5:
            return jsonify({"error": "Not enough sentences for reliable training."}), 400

        X_train, X_test, y_train, y_test = train_test_split(all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        if not X_train or not X_test: return jsonify({"error": "Failed to create train/test sets."}), 400

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_mixed_for_authorship, # Uses updated tokenizer
            token_pattern=None, ngram_range=(1, 2), max_features=10000, stop_words=_ALL_SW
        )
        X_train_vec, X_test_vec = vectorizer.fit_transform(X_train), vectorizer.transform(X_test)
        
        clf = MultinomialNB(); clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec); acc = accuracy_score(y_test, preds)
        
        report_target_names = sorted(list(clf.classes_))
        report = sk_classification_report(y_test, preds, digits=3, zero_division=0, target_names=report_target_names)
        
        distinctive_words = {}
        feature_names = vectorizer.get_feature_names_out()
        for idx, author_label_from_model in enumerate(clf.classes_):
            log_probs = clf.feature_log_prob_[idx]
            top_indices = log_probs.argsort()[-10:][::-1]
            distinctive_words[author_label_from_model] = [feature_names[i] for i in top_indices]

        sample_predictions = []
        for sent, true_l, pred_l in zip(X_test[:min(5, len(X_test))], y_test[:min(5, len(y_test))], preds[:min(5, len(preds))]):
            sample_predictions.append({
                "sentence_snippet": (sent[:100] + "…") if len(sent) > 100 else sent,
                "true_label": true_l, "predicted_label": pred_l
            })
        
        return jsonify({
            "accuracy": f"{acc:.3f}", "classification_report": report,
            "distinctive_words": distinctive_words, "sample_predictions": sample_predictions,
            "training_samples_count": len(X_train), "test_samples_count": len(X_test)
        })
    except ValueError as ve: return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Authorship analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Unexpected server error in authorship analysis."}), 500

# --- Main Execution ---
if __name__ == "__main__":
    print("Flask development server starting...")
    print("Verifying NLTK resources...")
    all_nltk_res_ok = True
    # Check if resources were actually downloaded and are findable by NLTK
    for res_key, res_path_val in REQUIRED_NLTK_RESOURCES.items():
        try: nltk.data.find(res_path_val) 
        except LookupError: # pragma: no cover
            print(f"Warning: NLTK resource '{res_key}' was not found initially or after download attempt. Functionality may be impaired.")
            all_nltk_res_ok = False # Mark that at least one resource is problematic
    
    if _ENG_SENT_TOKENIZER is None and "punkt" in REQUIRED_NLTK_RESOURCES : # Check if specific pre-load failed
        print("CRITICAL WARNING: Standard English sentence tokenizer (_ENG_SENT_TOKENIZER) failed to load at startup. Check 'punkt' NLTK resource.")
        all_nltk_res_ok = False


    if all_nltk_res_ok: print("All required NLTK resources appear to be available and essential tokenizers loaded.")
    else: print("WARNING: Some NLTK resources may be missing or failed to load. Application functionality might be impaired.")

    if not _TAGGER: print("Warning: Fugashi Tagger (Japanese) not initialized.")
    
    print("Starting Flask app server...")
    app.run(debug=True, port=8080, host='0.0.0.0') # User's preferred port