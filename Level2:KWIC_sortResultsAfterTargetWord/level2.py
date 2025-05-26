# ライブラリのインポート
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# 必要なNLTKリソースをダウンロード
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# ANSIカラーコード（コンソール出力用）
RED = '\033[31m'
BLUE = '\033[34m'
RESET = '\033[0m'

# Wikipediaページから本文を取得
def get_clean_wikipedia_text(url):
    try:
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'sup', 'table']):
            tag.decompose()
        text = soup.get_text()
        text = re.sub(r'\[[0-9]+\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

# メイン処理
def find_following_words(text, target_words, mode):
    if not text:
        print("No text to process.")
        return

    tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    target_str = ' '.join(target_words).lower()

    following_words = []
    contexts = []

    for i in range(len(tokens) - len(target_words)):
        if ' '.join(tokens[i:i+len(target_words)]) == target_str:
            next_idx = i + len(target_words)
            while next_idx < len(tokens):
                next_token = tokens[next_idx]
                if next_token:
                    following_words.append(next_token)
                    before = tokens[max(0, i-5):i]
                    after = tokens[next_idx+1:next_idx+6]
                    before_str = ' '.join(before)
                    after_str = ' '.join(after)
                    contexts.append((before_str, target_str, next_token, after_str))
                    break
                next_idx += 1

    if not following_words:
        print("No matches found for the target word(s).")
        return

    token_counts = Counter(following_words)
    sorted_tokens = [token for token, _ in token_counts.most_common()]

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
    else:
        print("Invalid mode. Please choose 1, 2, or 3.")

# ユーザー入力部分
def main():
    url = input("Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Banana): ").strip()
    text = get_clean_wikipedia_text(url)
    if not text:
        return

    target = input("Enter 1-2 word target (e.g., 'banana'): ").strip()
    target_words = target.split()
    if not 1 <= len(target_words) <= 2:
        print("Error: Enter 1 or 2 words.")
        return

    mode = input("Display mode - 1:Sequential 2:Token 3:POS: ").strip()
    find_following_words(text, target_words, mode)

# 実行
if __name__ == "__main__":
    main()