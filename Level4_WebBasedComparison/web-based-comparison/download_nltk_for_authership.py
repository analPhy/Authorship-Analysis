# download_nltk_for_authorship.py
import nltk
import sys
import os

# Render上でのプロジェクトルートを明示的に指定
project_root = os.environ.get('RENDER_PROJECT_ROOT', '/opt/render/project/src')
nltk_data_dir = os.path.join(project_root, 'nltk_data_on_render') # 固定のパス名

# ディレクトリが存在しない場合は作成
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
        print(f"--- [BUILD STEP] Created directory: {nltk_data_dir} ---")
    except OSError as e:
        print(f"--- [BUILD STEP] ERROR creating directory {nltk_data_dir}: {e} ---")
        sys.exit(1) # ディレクトリ作成失敗ならビルドエラー

# NLTK_DATA環境変数をこのスクリプト内で設定し、nltk.data.pathにも追加
# (Renderの環境変数設定と合わせて二重になる可能性もあるが、確実性を期すため)
os.environ['NLTK_DATA'] = nltk_data_dir
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir) # 検索パスの先頭に追加

print(f'--- [BUILD STEP] Starting NLTK Resource Download for Authorship to {nltk_data_dir} ---')
print(f'--- [BUILD STEP] NLTK search paths: {nltk.data.path} ---')

# 著者分析で最低限必要なNLTKリソース
# (キー: nltk.download()で使うID, 値: nltk.data.find()で使う検索パスの末尾)
resources_to_download = {
    'punkt': 'tokenizers/punkt.zip', # word_tokenize が依存する可能性 (展開後は punkt ディレクトリ)
    'words': 'corpora/words.zip'    # stopwordsiso (stopwords("en")) が依存 (展開後は words ファイル群)
}

all_downloads_successful = True
for res_id, res_find_path_suffix in resources_to_download.items():
    # nltk.data.find() は通常、展開後のディレクトリ名や主要ファイル名を期待する
    # .zip サフィックスはfind時には不要
    check_path = res_find_path_suffix.replace('.zip', '') 

    print(f'--- [BUILD STEP] Processing NLTK resource: {res_id} (expected at {check_path}) ---')
    try:
        nltk.data.find(check_path)
        print(f'--- [BUILD STEP] Resource {res_id} (for {check_path}) already exists. Skipping download. ---')
    except LookupError:
        print(f'--- [BUILD STEP] Resource {res_id} (for {check_path}) NOT FOUND. Attempting download... ---')
        try:
            if nltk.download(res_id, download_dir=nltk_data_dir, quiet=False, raise_on_error=True):
                print(f'--- [BUILD STEP] Successfully downloaded {res_id}. Verifying... ---')
                nltk.data.find(check_path) # ダウンロード後に再度検証
                print(f'--- [BUILD STEP] Resource {res_id} (for {check_path}) VERIFIED after download. ---')
            else:
                # raise_on_error=True のため、通常ここには到達しない
                print(f'--- [BUILD STEP] ERROR: nltk.download({res_id}) returned False. ---')
                all_downloads_successful = False
        except Exception as e:
            print(f'--- [BUILD STEP] ERROR downloading or verifying {res_id}: {e} ---')
            all_downloads_successful = False
            
print('--- [BUILD STEP] NLTK Resource Download for Authorship Finished ---')
if not all_downloads_successful:
    print('--- [BUILD STEP] CRITICAL BUILD FAILURE: One or more NLTK resources for authorship failed to download or verify. ---')
    sys.exit(1) # ひとつでも失敗したらビルドを失敗させる
else:
    print('--- [BUILD STEP] All specified NLTK resources for authorship successfully downloaded and verified during build. ---')