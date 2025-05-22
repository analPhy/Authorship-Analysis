# %% インポートとセットアップ
import re
import sys
import urllib.request
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# --- 多言語対応用ライブラリ ---
from langdetect import detect  # 言語判定
import fugashi                 # 日本語形態素解析
from stopwordsiso import stopwords  # 多言語ストップワード

# NLTK の英語用モデルが存在するか確認し、なければダウンロード
for res in ["punkt"]:
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res)

# fugashi 用の Tagger を初期化
_TAGGER = fugashi.Tagger()

# 日本語・英語ストップワードを結合（list にして scikit‑learn に渡す）
_EN_SW = stopwords("en")
_JA_SW = stopwords("ja")
_ALL_SW = sorted(_EN_SW.union(_JA_SW))

# 文分割用正規表現（日本語「。」と英語句読点を両方サポート）
_SENT_RE = re.compile(r"(?<=。)|(?<=[.!?])\s+")

# %% ユーティリティ関数

def fetch_wikipedia_text(url: str) -> str:
    """Download and clean the main text from a Wikipedia page."""
    try:
        html = urllib.request.urlopen(url).read()
    except Exception as err:
        sys.exit(f"Error fetching URL {url}: {err}")

    soup = BeautifulSoup(html, "html.parser")

    # 不要な要素を削除
    for tag in soup(["script", "style", "sup", "table"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = re.sub(r"\[\d+\]", "", text)  # 参照番号を削除
    text = re.sub(r"\s+", " ", text).strip()  # 連続する空白をまとめる
    return text


def mixed_sentence_tokenize(text: str):
    """日本語（。) と英語 (.!?) の両方で文を分割して返す。"""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def tokenize_mixed(text: str):
    """言語を判定して適切なトークナイザーで分割したトークンを返す。"""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"  # 判定失敗時は英語扱い

    if lang == "ja":
        return [tok.surface for tok in _TAGGER(text)]
    else:
        return word_tokenize(text)


def build_sentence_dataset(text: str, author_label: str, min_len: int = 30):
    """Return a list of sentences ≥ *min_len* characters tagged with *author_label*."""
    sentences = mixed_sentence_tokenize(text)
    filtered = [s for s in sentences if len(s) >= min_len]
    labels = [author_label] * len(filtered)
    return filtered, labels

# %% メイン処理

def main():
    print("=== Authorship Attribution Case Study (JA / EN) ===")
    url_a = input("Wikipedia URL for Author A: ").strip()
    url_b = input("Wikipedia URL for Author B: ").strip()

    print("\nDownloading Wikipedia pages…")
    text_a = fetch_wikipedia_text(url_a)
    text_b = fetch_wikipedia_text(url_b)

    if not text_a or not text_b:
        sys.exit("Failed to fetch one or both Wikipedia pages.")

    print("Splitting into sentences & labelling…")
    sentences_a, labels_a = build_sentence_dataset(text_a, "AuthorA")
    sentences_b, labels_b = build_sentence_dataset(text_b, "AuthorB")

    all_sentences = sentences_a + sentences_b
    all_labels = labels_a + labels_b

    X_train, X_test, y_train, y_test = train_test_split(
        all_sentences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print(f"Training samples: {len(X_train)} — Test samples: {len(X_test)}")

    # TF‑IDF（1-gram と 2-gram）でベクトル化（多言語対応）
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_mixed,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=20_000,
        stop_words=_ALL_SW,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 分類器を学習
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # 各著者に特徴的な単語を表示
    print("\n=== Distinctive Words/N‑grams per Author (Training Data) ===")
    feature_names = vectorizer.get_feature_names_out()
    for idx, author in enumerate(clf.classes_):
        log_probs = clf.feature_log_prob_[idx]
        top_indices = log_probs.argsort()[-15:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        print(f"\n{author}:\n  " + ", ".join(top_terms))

    # テストセットで評価
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print("\n=== Test‑set Performance ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, digits=3))

    # 予測例を表示
    print("Sample predictions on test sentences:\n")
    for sent, true_label, pred_label in zip(X_test[:10], y_test[:10], preds[:10]):
        snippet = (sent[:120] + "…") if len(sent) > 120 else sent
        print(f"• '{snippet}'\n   True: {true_label} | Predicted: {pred_label}\n")

if __name__ == "__main__":
    main()
