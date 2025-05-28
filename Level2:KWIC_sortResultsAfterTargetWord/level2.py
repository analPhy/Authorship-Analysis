# level2.py

# --- Imports ---
# EN: Import necessary libraries for text processing and web scraping
# JP: テキスト処理とウェブスクレイピングに必要なライブラリをインポート
import re
import urllib.request
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import fugashi
from langdetect import detect as lang_detect
import logging
import sys
from urllib.parse import urlparse

# --- ANSI Color Codes for Console Output ---
# EN: Define colors for highlighting target and following words
# JP: ターゲット単語と後続単語のハイライト用カラーコード
RED = '\033[31m'
BLUE = '\033[34m'
RESET = '\033[0m'

# --- Logging Setup ---
# EN: Configure logging to match app.py
# JP: app.pyと一致するロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout
)

# --- NLTK Resource Check ---
# EN: Ensure required NLTK resources are available
# JP: 必要なNLTKリソースが利用可能であることを確認
REQUIRED_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger"
}

logging.info("Checking NLTK resources...")
for download_key, path_to_find in REQUIRED_NLTK_RESOURCES.items():
    try:
        nltk.data.find(path_to_find)
        logging.info(f"NLTK resource '{download_key}' found.")
    except LookupError:
        logging.warning(f"NLTK resource '{download_key}' not found. Attempting to download...")
        try:
            nltk.download(download_key, quiet=False)
            nltk.data.find(path_to_find)
            logging.info(f"NLTK resource '{download_key}' downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download NLTK resource '{download_key}': {e}")
            print(f"Warning: NLTK resource '{download_key}' could not be downloaded. Some features may be impaired.")

# --- Initialize Fugashi Tagger ---
# EN: Set up Japanese tokenizer
# JP: 日本語トークナイザーの初期化
TAGGER = None
try:
    TAGGER = fugashi.Tagger()
    logging.info("fugashi Tagger initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize fugashi Tagger: {e}")
    print("Warning: Japanese tokenization will be limited.")

# --- Text Fetching and Cleaning (aligned with app.py) ---
# EN: Fetch and clean text from a URL, compatible with app.py's get_text_from_url_for_kwic
# JP: app.pyのget_text_from_url_for_kwicと互換性のあるテキスト取得・クリーンアップ
def get_clean_web_text(url):
    logging.info(f"Fetching URL: {url}")
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logging.warning(f"Invalid URL scheme: {url}")
            print(f"Error: Invalid URL scheme. Only http or https are allowed.")
            return None
        if not parsed_url.hostname:
            logging.warning(f"Invalid URL hostname: {url}")
            print("Error: Invalid URL hostname.")
            return None
        hostname_lower = parsed_url.hostname.lower()
        if not (hostname_lower == 'wikipedia.org' or hostname_lower.endswith('.wikipedia.org')):
            logging.warning(f"URL not from wikipedia.org: {url}")
            print(f"Error: URL '{url}' is not from wikipedia.org. Only Wikipedia URLs are supported.")
            return None

        # Fetch URL with same headers as app.py
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
                print(f"Error: The provided URL is not an HTML page (Content-Type: {content_type}).")
                return None
            html = response.read()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # Remove unnecessary tags (same as app.py)
        for tag in soup(['script', 'style', 'sup', 'table', 'head', 'link', 'meta', 'noscript',
                         'nav', 'footer', 'aside', 'form', 'input', 'button', 'img',
                         'audio', 'video', 'iframe', 'object', 'embed', 'header', 'svg', 'canvas']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        # Clean text (same as app.py's clean_text_for_kwic)
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        logging.info(f"Successfully fetched and cleaned text from URL: {url}")
        return text
    except (URLError, HTTPError) as e:
        logging.error(f"URL Error fetching {url}: {e.reason}")
        print(f"Error: Could not access the URL. Reason: {e.reason}")
        return None
    except TimeoutError:
        logging.error(f"Timeout fetching {url} after 15 seconds.")
        print("Error: URL fetching timed out.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {e}")
        print(f"Error: An unexpected error occurred while fetching the URL: {e}")
        return None

# --- Tokenization (aligned with app.py) ---
# EN: Tokenize text based on language, compatible with app.py's tokenize_mixed_for_authorship
# JP: app.pyのtokenize_mixed_for_authorshipと互換性のある言語ベースのトークン化
def tokenize_mixed(text, lang="en"):
    PUNCT_SKIP = {".", ",", "(", ")", "'"}
    try:
        if lang == "ja" and TAGGER:
            tokens = [tok.surface for tok in TAGGER(text) if tok.feature.pos1 not in ['助詞', '助動詞', '記号']]
        else:
            tokens = word_tokenize(text)
        # Remove punctuation-only tokens
        tokens = [t for t in tokens if t not in PUNCT_SKIP]
        return tokens
    except Exception as e:
        logging.error(f"Tokenization error: {e}")
        print(f"Error: Tokenization failed: {e}")
        return text.split()  # Fallback to simple split

# --- KWIC Search Logic ---
# EN: Find words following the target phrase and display in specified mode
# JP: ターゲットフレーズの後続単語を見つけ、指定されたモードで表示
def find_following_words(text, target_words, mode, context_window=5):
    if not text:
        print("Error: No text to process.")
        return

    # Detect language
    try:
        lang = lang_detect(text) if text else "en"
    except Exception:
        lang = "en"
    logging.info(f"Detected language: {lang}")

    # Tokenize text
    try:
        tokens_original = tokenize_mixed(text, lang)
        tokens_lower = [t.lower() for t in tokens_original]
    except Exception as e:
        logging.error(f"Tokenization failed: {e}")
        print("Error: Failed to tokenize the text.")
        return

    target_str = ' '.join(word.lower() for word in target_words)
    target_token_list = target_str.split()
    num_target_tokens = len(target_token_list)

    following_words = []
    contexts = []

    # Find matches and collect following words
    for i in range(len(tokens_lower) - num_target_tokens + 1):
        if tokens_lower[i:i + num_target_tokens] == target_token_list:
            next_idx = i + num_target_tokens
            if next_idx < len(tokens_lower):
                next_token = tokens_original[next_idx]  # Use original case for display
                before = tokens_original[max(0, i - context_window):i]
                after = tokens_original[next_idx + 1:next_idx + 1 + context_window]
                before_str = ' '.join(before)
                after_str = ' '.join(after)
                matched_str = ' '.join(tokens_original[i:i + num_target_tokens])
                following_words.append(next_token)
                contexts.append((before_str, matched_str, next_token, after_str))

    if not following_words:
        print(f"No matches found for the target phrase '{target_str}'.")
        return

    token_counts = Counter(following_words)
    sorted_tokens = [token for token, _ in token_counts.most_common()]

    # Display results based on mode
    if mode == '1':
        print("\nSequential Results (sorted by frequency):")
        for token in sorted_tokens:
            for before, target, next_token, after in contexts:
                if next_token == token:
                    print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after}")
    elif mode == '2':
        print("\nMost Frequent Tokens:")
        for token in sorted_tokens:
            print(f"{token}: {token_counts[token]}")
            for before, target, next_token, after in contexts:
                if next_token == token:
                    print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after}")
    elif mode == '3':
        print("\nMost Frequent POS Tags (sorted by token frequency):")
        try:
            if lang == "ja" and TAGGER:
                tagged_tokens = [(tok.surface, tok.feature.pos1) for tok in TAGGER(' '.join(following_words))]
            else:
                tagged_tokens = pos_tag(following_words)
            token_to_pos = dict(tagged_tokens)
            sorted_tagged = sorted(tagged_tokens, key=lambda x: token_counts[x[0]], reverse=True)
            seen = set()
            for word, tag in sorted_tagged:
                if word not in seen:
                    seen.add(word)
                    print(f"{tag} ({word}): {token_counts[word]}")
                    for before, target, next_token, after in contexts:
                        if next_token == word:
                            print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after}")
        except Exception as e:
            logging.error(f"POS tagging failed: {e}")
            print("Error: POS tagging failed. Displaying tokens without tags.")
            for token in sorted_tokens:
                print(f"{token}: {token_counts[token]}")
                for before, target, next_token, after in contexts:
                    if next_token == token:
                        print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after}")
    else:
        print("Error: Invalid mode. Please choose 1 (Sequential), 2 (Token), or 3 (POS).")

# --- Main Function ---
# EN: Handle user input and execute the search
# JP: ユーザー入力を受け取り、検索を実行
def main():
    # Get URL
    url = input("Enter Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Banana): ").strip()
    if not url:
        print("Error: Please provide a Wikipedia URL.")
        return

    # Fetch and clean text
    text = get_clean_web_text(url)
    if not text:
        return

    # Get target phrase
    target = input("Enter 1-2 word target (e.g., 'banana' or 'バナナ'): ").strip()
    target_words = target.split()
    if not 1 <= len(target_words) <= 2:
        print("Error: Please enter 1 or 2 words.")
        return

    # Get display mode
    mode = input("Display mode - 1:Sequential 2:Token 3:POS: ").strip()
    if mode not in ['1', '2', '3']:
        print("Error: Invalid mode. Please choose 1, 2, or 3.")
        return

    # Execute search
    find_following_words(text, target_words, mode)

# --- Execution ---
if __name__ == "__main__":
    print("Starting Wikipedia Phrase Search...")
    main()