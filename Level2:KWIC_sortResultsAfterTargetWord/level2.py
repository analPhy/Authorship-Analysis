# 必要なライブラリをインストール
!pip install beautifulsoup4
!pip install nltk

# ライブラリのインポート
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# NLTKリソースのダウンロード
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ANSIエスケープコードで色を定義
RED = '\033[31m'
BLUE = '\033[34m'
RESET = '\033[0m'

# --- テキスト取得 ---
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

# --- メイン処理 ---
def find_following_words(text, target_words, mode):
    if not text:
        print("No text to process.")
        return

    # トークン化
    tokens = [t for t in word_tokenize(text.lower()) if t]  # 空トークンを除外、大文字小文字を統一
    target_str = ' '.join(target_words).lower()

    following_words = []
    contexts = []  # (前5トークン, ターゲット, 後続トークン, 後5トークン)のリスト

    # ターゲット語の後続単語を収集（空白や句読点をスキップ）
    for i in range(len(tokens) - len(target_words)):
        if ' '.join(tokens[i:i+len(target_words)]).lower() == target_str:
            next_idx = i + len(target_words)
            while next_idx < len(tokens):
                next_token = tokens[next_idx]
                if next_token and next_token.isalnum():
                    following_words.append(next_token)
                    # 前5トークンと後5トークンを取得
                    before = tokens[max(0, i-5):i]
                    after = tokens[next_idx:next_idx+5]
                    before_str = ' '.join(before) if before else ''
                    after_str = ' '.join(after) if after else ''
                    contexts.append((before_str, target_str, next_token, after_str))
                    break
                next_idx += 1

    if not following_words:
        print("No matches found for the target word(s).")
        return

    # モードに応じた結果表示
    if mode == '1':
        print("\nSequential Results:")
        for before, target, next_token, after in contexts:
            print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after[len(next_token):].strip()}")
    elif mode == '2':
        print("\nMost Frequent Tokens:")
        token_counts = Counter(following_words)
        for token, freq in token_counts.most_common():
            print(f"{token}: {freq}")
            for before, target, next_token, after in contexts:
                if next_token == token:
                    after_tokens = after.split()[:4]
                    after_str = ' '.join(after_tokens)
                    print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after_str[len(next_token):].strip()}")
    elif mode == '3':
        print("\nMost Frequent POS Tags:")
        # 青文字ワード（next_token）の品詞を個別にカウント
        pos_tags = [pos_tag([next_token])[0][1] for _, _, next_token, _ in contexts]
        pos_counts = Counter(pos_tags)
        for pos, freq in pos_counts.most_common():
            print(f"{pos}: {freq}")
            for before, target, next_token, after in contexts:
                if pos_tag([next_token])[0][1] == pos:
                    print(f"{before} {RED}{target}{RESET} {BLUE}{next_token}{RESET} {after[len(next_token):].strip()}")
    else:
        print("Invalid mode. Please choose 1, 2, or 3.")

# --- ユーザー入力 ---
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
