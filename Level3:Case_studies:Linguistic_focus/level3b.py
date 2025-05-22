# %% インポートとセットアップ
import re
import sys
import urllib.request
from collections import Counter

from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# NLTK データが存在するか確認し、なければダウンロード
for res in ["punkt"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res)

# %% ユーティリティ関数

def fetch_wikipedia_text(url: str) -> str:
    """Wikipedia ページから本文をダウンロードし、整形して返す。"""
    try:
        html = urllib.request.urlopen(url).read()
    except Exception as err:
        sys.exit(f"URL {url} の取得中にエラーが発生しました: {err}")

    soup = BeautifulSoup(html, "html.parser")

    # スクリプト、スタイル、参照表など不要な要素を削除
    for tag in soup(["script", "style", "sup", "table"]):
        tag.decompose()

    text = soup.get_text(" ")
    # 参照番号 [1], [2] … を削除
    text = re.sub(r"\[\d+\]", "", text)
    # 連続する空白を 1 つにまとめる
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_sentence_dataset(text: str, author_label: str, min_len: int = 30):
    """文章長が *min_len* 以上の文を抽出し、*author_label* でタグ付けして返す。"""
    sentences = sent_tokenize(text)
    filtered = [s.strip() for s in sentences if len(s.strip()) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels


# %% メインルーチン

def main():
    print("=== Authorship Attribution Case Study ===")
    url_a = input("Wikipedia URL for Author A: ").strip()
    url_b = input("Wikipedia URL for Author B: ").strip()

    print("\nWikipedia ページをダウンロードしています…")
    text_a = fetch_wikipedia_text(url_a)
    text_b = fetch_wikipedia_text(url_b)

    if not text_a or not text_b:
        sys.exit("Wikipedia ページの取得に失敗しました。")

    print("文を分割してラベル付けしています…")
    sentences_a, labels_a = build_sentence_dataset(text_a, "AuthorA")
    sentences_b, labels_b = build_sentence_dataset(text_b, "AuthorB")

    all_sentences = sentences_a + sentences_b
    all_labels = labels_a + labels_b

    X_train, X_test, y_train, y_test = train_test_split(
        all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print(f"学習サンプル: {len(X_train)} — テストサンプル: {len(X_test)}")

    # TF‑IDF（1-gram と 2-gram）でベクトル化
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 分類器を学習
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # 各著者に特徴的な単語を表示（対数確率上位 15 件）
    print("\n=== 学習データにおける著者別特徴語・N‑gram ===")
    feature_names = vectorizer.get_feature_names_out()
    for idx, author in enumerate(clf.classes_):
        log_probs = clf.feature_log_prob_[idx]
        top_indices = log_probs.argsort()[-15:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        print(f"\n{author}:\n  " + ", ".join(top_terms))

    # 20 % のテストセットで評価
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print("\n=== テストセットでの性能 ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, digits=3))

    # 予測例をいくつか表示
    print("テスト文に対する予測例:\n")
    for sent, true_label, pred_label in zip(X_test[:10], y_test[:10], preds[:10]):
        snippet = (sent[:120] + "…") if len(sent) > 120 else sent
        print(f"• '{snippet}'\n   正解: {true_label} | 予測: {pred_label}\n")

    # 任意入力による対話的分類
    while True:
        user_txt = input("著者を推定したい文を入力してください（空行で終了）: ").strip()
        if not user_txt:
            break
        vec = vectorizer.transform([user_txt])
        guess = clf.predict(vec)[0]
        probs = clf.predict_proba(vec)[0]
        print(f"推定結果: {guess} (信頼度 {probs[clf.classes_.tolist().index(guess)]:.2f})\n")


if __name__ == "__main__":
    main()
