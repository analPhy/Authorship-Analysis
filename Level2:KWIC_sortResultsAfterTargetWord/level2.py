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
nltk.download('averaged_perceptron_tagger_eng')  # Added to resolve LookupError

# --- テキスト取得 ---
def get_clean_wikipedia_text(url):
    try:
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')
        # 不要なタグを削除
        for tag in soup(['script', 'style', 'sup', 'table']):
            tag.decompose()
        text = soup.get_text()
        # 参照番号や余分な空白を削除
        text = re.sub(r'\[[0-9]+\]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

# --- メイン処理 ---
def find_following_words(text, target_words, mode):
    if not text:
        print("No text to process.")
        return

    # トークン化とPOSタグ付け
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    following_words = []

    # ターゲット語の後続単語を収集
    for i in range(len(tokens) - len(target_words)):
        if tokens[i:i+len(target_words)] == target_words:
            if i + len(target_words) < len(tokens):
                following_words.append(tokens[i + len(target_words)])

    if not following_words:
        print("No matches found for the target word(s).")
        return

    # モードに応じた結果表示
    if mode == '1':
        print("\nSequential Results:")
        for word in following_words:
            print(word)
    elif mode == '2':
        print("\nMost Frequent Tokens:")
        for word, freq in Counter(following_words).most_common():
            print(f"{word}: {freq}")
    elif mode == '3':
        print("\nMost Frequent POS Tags:")
        pos_tags = [pos for word, pos in pos_tag(following_words)]
        for tag, freq in Counter(pos_tags).most_common():
            print(f"{tag}: {freq}")
    else:
        print("Invalid mode. Please choose 1, 2, or 3.")

# --- ユーザー入力 ---
def main():
    # Wikipedia URLの入力
    url = input("Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Python_(programming_language)): ").strip()
    text = get_clean_wikipedia_text(url)
    if not text:
        return

    # ターゲット語の入力
    target = input("Enter 1-2 word target (case sensitive, e.g., 'machine learning'): ").strip()
    target_words = target.split()
    if not 1 <= len(target_words) <= 2:
        print("Error: Enter 1 or 2 words.")
        return

    # 表示モードの入力
    mode = input("Display mode - 1:Sequential 2:Token 3:POS: ").strip()
    find_following_words(text, target_words, mode)

# 実行
if __name__ == "__main__":
    main()
    