import nltk
import os

# NLTKのデフォルトデータパスにダウンロードする
# (もし特定のパスにダウンロードしたい場合は nltk.download('punkt_tab', download_dir='/custom/path'))
print(f"NLTK data path: {nltk.data.path}")
try:
    print("Attempting to download 'punkt_tab'...")
    nltk.download('punkt_tab')
    print("'punkt_tab' download command executed.")
except Exception as e:
    print(f"Error downloading 'punkt_tab': {e}")

# ダウンロード後、期待されるパスが存在するか確認
# nltk.data.path の最初の有効なパスを使うことが多い
expected_path_base = os.path.join(nltk.data.path[0], "tokenizers", "punkt_tab")
expected_path_english = os.path.join(expected_path_base, "english") # エラーログが探しているパス

print(f"Checking for base directory: {expected_path_base}")
if os.path.exists(expected_path_base) and os.path.isdir(expected_path_base):
    print(f"Base directory '{expected_path_base}' exists.")
    print(f"Contents of base directory: {os.listdir(expected_path_base)}")
    
    print(f"Checking for English subdirectory: {expected_path_english}")
    if os.path.exists(expected_path_english) and os.path.isdir(expected_path_english):
        print(f"English subdirectory '{expected_path_english}' exists.")
        print(f"Contents of English subdirectory: {os.listdir(expected_path_english)}")
    else:
        print(f"English subdirectory '{expected_path_english}' NOT found.")
else:
    print(f"Base directory '{expected_path_base}' NOT found.")

# 試しにPunktTokenizerを初期化
try:
    print("Attempting to initialize PunktTokenizer(language='english')...")
    tokenizer = nltk.tokenize.PunktTokenizer(language='english')
    print("PunktTokenizer initialized successfully.")
except Exception as e:
    print(f"Error initializing PunktTokenizer: {e}")