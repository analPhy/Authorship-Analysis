# download_nltk.py の冒頭部分
import nltk
import sys
import os

# Render上でのプロジェクトルートを基点とする
# 環境変数 RENDER_PROJECT_ROOT があればそれを使う (Renderが提供しているか要確認)
# なければ、一般的な /opt/render/project/src/ を想定
project_root = os.environ.get('RENDER_PROJECT_ROOT', '/opt/render/project/src')
nltk_data_dir = os.path.join(project_root, 'nltk_data_on_render') # プロジェクトルート直下に作成

# ディレクトリが存在しない場合は作成
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
        print(f"--- [BUILD STEP] Created directory: {nltk_data_dir} ---")
    except OSError as e:
        print(f"--- [BUILD STEP] ERROR creating directory {nltk_data_dir}: {e} ---")
        sys.exit(1)

os.environ['NLTK_DATA_SCRIPT_SET'] = nltk_data_dir # スクリプト内で設定したことを示す環境変数 (デバッグ用)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

# ... (以降のダウンロード処理は変更なし) ...