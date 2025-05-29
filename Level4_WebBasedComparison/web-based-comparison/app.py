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
import string # For punctuation set

# --- Authorship Attribution Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as sk_classification_report, accuracy_score

# --- Multilingual Support Libraries ---
from langdetect import detect as lang_detect
import fugashi
from stopwordsiso import stopwords

# --- Punctuation Handling ---
PUNCTUATION_SET = set(string.punctuation + '。、「」』『【】・（）　')

def remove_punctuation_from_token(token: str) -> str:
    """Removes all defined punctuation characters from a single token."""
    return ''.join(char for char in token if char not in PUNCTUATION_SET)

def preprocess_tokens(tokens_list: list[str]) -> list[str]:
    """Removes punctuation from each token and filters out any empty tokens."""
    if not isinstance(tokens_list, list): # Ensure input is a list
        logging.warning(f"preprocess_tokens received non-list input: {type(tokens_list)}")
        return []
    processed_tokens = [remove_punctuation_from_token(str(token)) for token in tokens_list] # Ensure token is str
    return [token for token in processed_tokens if token]


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
    "words": "corpora/words",
    # "punkt_tab": "tokenizers/punkt_tab", # Usually punkt is sufficient. Kept if user had specific reason.
}

logging.info("Checking NLTK resources...")
for download_key, path_to_find in REQUIRED_NLTK_RESOURCES.items():
    try:
        nltk.data.find(path_to_find)
        logging.info(f"NLTK resource '{download_key}' (for path '{path_to_find}') found.")
    except LookupError:
        logging.warning(f"NLTK resource '{download_key}' (for path '{path_to_find}') not found. Attempting to download '{download_key}'...")
        try:
            download_successful = nltk.download(download_key, quiet=False)
            if download_successful is None or download_successful: # nltk.download can return None or True on success
                logging.info(f"NLTK download command for '{download_key}' executed.")
                nltk.data.find(path_to_find) # Verify after download
                logging.info(f"NLTK resource '{download_key}' now found after download.")
            else:
                logging.error(f"NLTK download for '{download_key}' returned False. Resource may not be available.")
                # Attempt to verify anyway, as it might have been downloaded by other means or existed.
                try:
                    nltk.data.find(path_to_find)
                    logging.info(f"NLTK resource '{download_key}' found despite download command returning False.")
                except LookupError:
                    logging.error(f"CRITICAL: NLTK resource '{download_key}' still not found after download command returned False. Manual intervention likely required.")
        except Exception as e_download:
            logging.error(f"Error during NLTK resource '{download_key}' download or verification: {e_download}")
            logging.error(f"Please check network connectivity and NLTK setup. You may need to run: python -m nltk.downloader {download_key}")

# --- Global Settings for Authorship Attribution ---
_TAGGER = None
try:
    _TAGGER = fugashi.Tagger()
    logging.info("fugashi Tagger initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize fugashi Tagger: {e}. Japanese tokenization for authorship will not work.")

_EN_SW, _JA_SW = set(), set()
try:
    _EN_SW = stopwords("en"); _JA_SW = stopwords("ja")
    logging.info("English and Japanese stopwords (for authorship) loaded.")
except Exception as e: logging.error(f"Failed to load stopwords (for authorship): {e}")
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

# === Text processing functions (as provided by user, with minor adjustments if necessary) ===
def get_text_from_url_for_kwic(url): # Renamed from user's get_text_from_url to be specific
    logging.info(f"Attempting to fetch URL (for KWIC search): {url}")
    try:
        req = urllib.request.Request(
            url, data=None,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()):
                 logging.warning(f"URL (KWIC search) not HTML: {url} (Content-Type: {content_type})")
                 raise ValueError(f"URL does not appear to be HTML. (Content-Type: {content_type})")
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                         'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                         'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully fetched and parsed URL (KWIC): {url}")
        return text
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching (KWIC) {url}: {e.reason}")
        raise ValueError(f"Could not access URL. Reason: {e.reason}")
    except TimeoutError:
         logging.error(f"Timeout fetching (KWIC) {url}.")
         raise ValueError(f"URL fetching timed out.")
    except ValueError as e:
          logging.error(f"Value Error processing (KWIC) {url}: {e}")
          raise
    except Exception as e:
        logging.error(f"Unexpected error in get_text_from_url_for_kwic for {url}: {e}\n{traceback.format_exc()}")
        raise ValueError(f"Unexpected error fetching/processing URL.")

def clean_text_for_kwic(text):
    text = re.sub(r'\[\d+\]', '', text) # Remove citation numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_wikipedia_text_for_authorship(url: str) -> str: # Assumed this is specific for authorship if logic differs
    logging.info(f"Fetching text for authorship from URL: {url}")
    # This function was identical to get_text_from_url_for_kwic in user's provided code,
    # except for User-Agent and some removed tags. Consolidating or keeping separate depends on need.
    # For now, keeping as is from user's code structure.
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Python-Flask-Authorship-App/1.0)'})
        with urllib.request.urlopen(req, timeout=20) as response:
            html = response.read()
            content_type = response.getheader('Content-Type')
            if not (content_type and 'text/html' in content_type.lower()):
                 raise ValueError(f"URL for authorship not HTML. (Content-Type: {content_type})")
    except (URLError, HTTPError) as e:
        raise ValueError(f"Could not access authorship URL. Reason: {e.reason}")
    except TimeoutError:
        raise ValueError("Authorship URL fetching timed out.")
    except Exception as err:
        raise ValueError(f"Unexpected error fetching authorship URL: {err}")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "sup", "table", "nav", "footer", "aside", "header", "form", "figure", "figcaption", "link", "meta", "input", "button", "img", "audio", "video", "iframe", "object", "embed", "svg", "canvas", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\[\d+\]", "", text) # Remove citation numbers
    text = re.sub(r"\s+", " ", text).strip()
    logging.info(f"Successfully parsed text for authorship from URL: {url}")
    return text

# --- Tokenizer for Authorship (integrates punctuation removal) ---
def tokenize_mixed_for_authorship(text: str) -> list[str]:
    initial_tokens = []
    if not _TAGGER:
        logging.warning("Fugashi Tagger not available, using NLTK word_tokenize for authorship.")
        try: initial_tokens = word_tokenize(text)
        except Exception as e_tok:
            logging.error(f"NLTK word_tokenize fallback failed in authorship: {e_tok}"); initial_tokens = text.split()
    else:
        try: lang = lang_detect(text)
        except Exception: logging.warning(f"Langdetect failed for authorship sample: '{text[:100]}'. Defaulting to English."); lang = "en"
        if lang == "ja": initial_tokens = [tok.surface for tok in _TAGGER(text)]
        else:
            try: initial_tokens = word_tokenize(text)
            except Exception as e_tok_en: logging.error(f"NLTK word_tokenize for non-JA failed in authorship: {e_tok_en}"); initial_tokens = text.split()
    
    punctuation_removed_tokens = preprocess_tokens(initial_tokens) # Uses global helper
    lowercased_tokens = [token.lower() for token in punctuation_removed_tokens]
    return lowercased_tokens

def mixed_sentence_tokenize_for_authorship(text: str): # This was an existing function in user's code for sentence splitting
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def build_sentence_dataset_for_authorship(text: str, author_label: str, min_len: int = 30):
    sentences = mixed_sentence_tokenize_for_authorship(text) # Sentence splitting
    # Note: tokenize_mixed_for_authorship is used by TfidfVectorizer, not directly here for sentences.
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

    if not url: return jsonify({"error": "Please provide a Web Page URL."}), 400
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https'] or not parsed_url.hostname:
             return jsonify({"error": "Invalid URL (scheme/hostname)."}), 400
        # Add is_restricted_hostname check here if general URLs are to be supported more safely
    except Exception as e: return jsonify({"error": f"Invalid URL format: {e}"}), 400
    if not query_input: return jsonify({"error": "Please provide a search query."}), 400

    try:
        raw_text = get_text_from_url_for_kwic(url)
        text_cleaned = clean_text_for_kwic(raw_text)
        if not text_cleaned: return jsonify({"results": [], "error": "Could not extract text."}), 200
        initial_tokens_from_page = word_tokenize(text_cleaned)
        words_from_page_processed = preprocess_tokens(initial_tokens_from_page)
        if not words_from_page_processed: return jsonify({"results": [], "error": "No searchable words after processing."}), 200
    except ValueError as e: return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"KWIC: Error processing URL/text: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Error processing URL content."}), 500

    results = []
    backend_context_window_size = 10

    if search_type == 'token':
        raw_target_tokens = query_input.split()
        target_tokens_processed = preprocess_tokens(raw_target_tokens)
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
                results.append({
                    "context_words": before + matched_segment + after,
                    "matched_start": len(before), "matched_end": len(before) + num_target_tokens,
                    "original_text_index": i # For stable sort if custom sort uses it
                })
    
    elif search_type == 'pos':
        target_pos_tag_query = query_input.strip().upper()
        if not target_pos_tag_query or " " in target_pos_tag_query: return jsonify({"error": "Invalid POS tag query."}), 400
        logging.info(f"POS tagging for: '{target_pos_tag_query}' on {len(words_from_page_processed)} tokens.")
        try:
            tagged_words = nltk.pos_tag(words_from_page_processed)
        except Exception as e_pos:
            logging.error(f"NLTK pos_tag FAILED: {e_pos}\n{traceback.format_exc()}")
            return jsonify({"error": "POS tagging failed on server."}), 500
        for i, (word, tag) in enumerate(tagged_words):
            if tag == target_pos_tag_query:
                before = words_from_page_processed[max(0, i - backend_context_window_size): i]
                after = words_from_page_processed[i + 1 : i + 1 + backend_context_window_size]
                results.append({
                    "context_words": before + [words_from_page_processed[i]] + after,
                    "matched_start": len(before), "matched_end": len(before) + 1,
                    "original_text_index": i # For stable sort
                })

    elif search_type == 'entity':
        target_entity_type_query = query_input.strip().upper()
        if not target_entity_type_query or " " in target_entity_type_query: return jsonify({"error": "Invalid Entity type query."}), 400
        logging.info(f"Entity recognition for: '{target_entity_type_query}' on {len(words_from_page_processed)} tokens.")
        try:
            tagged_words_for_ner = nltk.pos_tag(words_from_page_processed)
            chunked_entities_tree = nltk.ne_chunk(tagged_words_for_ner)
            iob_tags = nltk.chunk.util.tree2conlltags(chunked_entities_tree)
        except Exception as e_ner:
            logging.error(f"NLTK NER FAILED: {e_ner}\n{traceback.format_exc()}")
            return jsonify({"error": "Entity recognition failed on server."}), 500
        idx = 0
        while idx < len(iob_tags):
            word_from_iob, _, iob_label = iob_tags[idx]
            if iob_label.startswith('B-') and iob_label[2:] == target_entity_type_query:
                current_entity_words = [word_from_iob]; entity_start_idx = idx; next_idx = idx + 1
                while next_idx < len(iob_tags) and iob_tags[next_idx][2].startswith('I-') and iob_tags[next_idx][2][2:] == target_entity_type_query:
                    current_entity_words.append(iob_tags[next_idx][0]); next_idx += 1
                before = words_from_page_processed[max(0, entity_start_idx - backend_context_window_size): entity_start_idx]
                after = words_from_page_processed[next_idx : next_idx + backend_context_window_size]
                results.append({
                    "context_words": before + current_entity_words + after,
                    "matched_start": len(before), "matched_end": len(before) + len(current_entity_words),
                    "original_text_index": entity_start_idx # For stable sort
                })
                idx = next_idx
            else: idx += 1
    else:
        return jsonify({"error": "Invalid search type."}), 400

    # --- User's custom sorting logic ---
    if results:
        logging.info(f"Applying custom sorting to {len(results)} results.")
        pattern_groups = {}
        for r_item in results: 
            kwic_ctx_words = r_item["context_words"]
            kwic_match_end = r_item["matched_end"]
            pattern_key = "" # Default for matches at the end of context
            if kwic_match_end < len(kwic_ctx_words):
                next_words_seg = kwic_ctx_words[kwic_match_end:min(kwic_match_end + 3, len(kwic_ctx_words))]
                pattern_key = " ".join(word.lower() for word in next_words_seg)
            if pattern_key not in pattern_groups: pattern_groups[pattern_key] = []
            pattern_groups[pattern_key].append(r_item)

        freq_and_pos_groups = {}
        if pattern_groups:
            for pattern, group in pattern_groups.items():
                freq = len(group)
                # Use original_text_index for stable sort key if available, else use a placeholder
                first_pos = min(item.get("original_text_index", float('inf')) for item in group)
                if freq not in freq_and_pos_groups: freq_and_pos_groups[freq] = []
                freq_and_pos_groups[freq].append((first_pos, pattern, group))
        
        sorted_results_final = []
        if freq_and_pos_groups:
            for freq_key in sorted(freq_and_pos_groups.keys(), reverse=True):
                groups_at_freq = sorted(freq_and_pos_groups[freq_key], key=lambda x: x[0]) 
                for _, _, group_items_list in groups_at_freq:
                    sorted_results_final.extend(group_items_list)
            results = sorted_results_final
            logging.info(f"Sorting applied. Result count: {len(results)}")
        else:
            logging.info("No patterns for custom sorting, using original find order (or lack thereof for non-token).")
            # For non-token types, results might not have original_text_index consistently yet.
            # If original_text_index is crucial, ensure it's added to POS/Entity results items too.
            # For now, if no groups, 'results' remains as found.
            # If results were from token search, they already have original_text_index.
            # We can add a simple sort by original_text_index if it exists as a fallback.
            if all("original_text_index" in r for r in results):
                 results.sort(key=lambda x: x.get("original_text_index", 0))


    if not results:
        display_query = ' '.join(target_tokens_processed) if search_type == 'token' and 'target_tokens_processed' in locals() and target_tokens_processed else query_input
        logging.info(f"Query '{query_input}' (Type: {search_type}, Processed: '{display_query}') not found.")
        return jsonify({"results": [], "error": f"The query '{display_query}' (Type: {search_type}) was not found."}), 200
        
    logging.info(f"Found {len(results)} occurrences for query '{query_input}' (Type: {search_type}).")
    return jsonify({"results": results})

# === Authorship Analysis Endpoint ( Largely as provided by user, with tokenizer update ) ===
@app.route('/api/authorship', methods=['POST'])
def authorship_analysis():
    data = request.json
    url_a, url_b = data.get('url_a', '').strip(), data.get('url_b', '').strip()
    logging.info(f"Authorship request: URL A: {url_a}, URL B: {url_b}")

    if not url_a or not url_b: return jsonify({"error": "Please provide two URLs."}), 400
    # URL validation (as per user code - specific to Wikipedia)
    for i, url_val in enumerate([url_a, url_b]):
        label = "A" if i == 0 else "B"
        try:
            parsed = urlparse(url_val)
            if not parsed.scheme in ['http', 'https'] or not parsed.hostname: raise ValueError("Invalid URL")
            hostname_lower = parsed.hostname.lower()
            # Keeping Wikipedia restriction as per user's original authorship code
            if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
                return jsonify({"error": f"URL for Source {label} ('{url_val}') is not from wikipedia.org."}), 400
        except ValueError: return jsonify({"error": f"Invalid URL for Source {label}: {url_val}"}), 400

    try:
        text_a = fetch_wikipedia_text_for_authorship(url_a) # Uses user's specific fetcher
        text_b = fetch_wikipedia_text_for_authorship(url_b)
        if not text_a or not text_b: return jsonify({"error": "Failed to fetch content."}), 500

        sentences_a, labels_a = build_sentence_dataset_for_authorship(text_a, "AuthorA")
        sentences_b, labels_b = build_sentence_dataset_for_authorship(text_b, "AuthorB")
        if not sentences_a or not sentences_b: return jsonify({"error": "Not enough valid sentences."}), 400

        all_sentences, all_labels = sentences_a + sentences_b, labels_a + labels_b
        if len(set(all_labels)) < 2: return jsonify({"error": "Need sentences from two distinct sources."}), 400
        
        MIN_SAMPLES = 2 
        if labels_a.count("AuthorA") < MIN_SAMPLES or labels_b.count("AuthorB") < MIN_SAMPLES or len(all_sentences) < 5:
            return jsonify({"error": "Not enough sentences for reliable training."}), 400

        X_train, X_test, y_train, y_test = train_test_split(all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        if not X_train or not X_test: return jsonify({"error": "Failed to create train/test sets."}), 400

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_mixed_for_authorship, # Using updated tokenizer
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
    for res_key, res_path_val in REQUIRED_NLTK_RESOURCES.items():
        try: nltk.data.find(res_path_val)
        except LookupError:
            print(f"Warning: NLTK resource '{res_key}' not found. Functionality may be impaired.")
            all_nltk_res_ok = False
    if all_nltk_res_ok: print("All required NLTK resources appear available.")
    if not _TAGGER: print("Warning: Fugashi Tagger (Japanese) not initialized.")
    
    print("Starting Flask app server...")
    app.run(debug=True, port=8080, host='0.0.0.0') # User's port