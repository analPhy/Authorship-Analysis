# download_nltk.py
import nltk
import sys
import os

# Render上でのプロジェクトルートを明示的に指定
project_root = '/opt/render/project/src' 
nltk_data_dir = os.path.join(project_root, 'nltk_data_on_render') # 固定のパス名に変更

# ディレクトリが存在しない場合は作成
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
        print(f"--- [BUILD STEP] Created directory: {nltk_data_dir} ---")
    except OSError as e:
        print(f"--- [BUILD STEP] ERROR creating directory {nltk_data_dir}: {e} ---")
        sys.exit(1)

# NLTK_DATA環境変数をこのスクリプト内で設定し、nltk.data.pathにも追加
os.environ['NLTK_DATA'] = nltk_data_dir
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir) # 検索パスの先頭に追加

# ... (以降のダウンロード処理は変更なし) ...