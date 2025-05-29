# download_nltk.py
import nltk
import sys
import os

# ダウンロード先をリポジトリのルートディレクトリ内の 'nltk_data_render' に設定
# Render上では /opt/render/project/src/ がリポジトリのルートになることが多い
# os.getcwd() が /opt/render/project/src/ を指すことを期待
# もし確実でなければ、絶対パス '/opt/render/project/src/nltk_data_render' を直接指定も可
# ただし、ローカルでのテストとRender上でのパスの違いに注意が必要になる場合がある
# ここでは、スクリプトがプロジェクトルートにある前提で相対パス的に構築
# script_dir = os.path.dirname(os.path.abspath(__file__)) # download_nltk.py がある場所
# nltk_data_dir = os.path.join(script_dir, 'nltk_data_render')
# Renderのビルド環境ではカレントディレクトリが /opt/render/project/src/ になっていることが多いので、これで試す
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data_render')


# ディレクトリが存在しない場合は作成
if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
        print(f"--- [BUILD STEP] Created directory: {nltk_data_dir} ---")
    except OSError as e:
        print(f"--- [BUILD STEP] ERROR creating directory {nltk_data_dir}: {e} ---")
        sys.exit(1) # ディレクトリ作成失敗ならビルドエラー

os.environ['NLTK_DATA'] = nltk_data_dir
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

print(f'--- [BUILD STEP] Starting NLTK Resource Download to {nltk_data_dir} ---')
print(f'--- [BUILD STEP] NLTK search paths: {nltk.data.path} ---')

resources_to_download = {
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'punkt': 'tokenizers/punkt',
    'words': 'corpora/words',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
    'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab'
}

all_downloads_successful = True
for res_id, res_check_path_key in resources_to_download.items():
    # nltk.data.find は .zip を除いたディレクトリ名やファイル名を期待することが多い
    # resources_to_download の値（res_check_path_key）をそのままチェックパスとして使用
    check_path = res_check_path_key

    print(f'--- [BUILD STEP] Processing NLTK resource: {res_id} (expected at {check_path} within nltk_data_dir) ---')
    try:
        nltk.data.find(check_path) # このチェックは nltk_data_dir 配下のパスに対して行われる
        print(f'--- [BUILD STEP] Resource {res_id} already exists. Skipping download. ---')
    except LookupError:
        print(f'--- [BUILD STEP] Resource {res_id} NOT FOUND. Attempting download... ---')
        try:
            if nltk.download(res_id, download_dir=nltk_data_dir, quiet=False, raise_on_error=True):
                print(f'--- [BUILD STEP] Successfully downloaded {res_id}. Verifying... ---')
                nltk.data.find(check_path)
                print(f'--- [BUILD STEP] Resource {res_id} VERIFIED after download. ---')
            else:
                print(f'--- [BUILD STEP] ERROR: nltk.download({res_id}) returned False. ---')
                all_downloads_successful = False
        except Exception as e:
            print(f'--- [BUILD STEP] ERROR downloading or verifying {res_id}: {e} ---')
            all_downloads_successful = False
            
print('--- [BUILD STEP] NLTK Resource Download Finished ---')
if not all_downloads_successful:
    print('--- [BUILD STEP] CRITICAL BUILD FAILURE: One or more NLTK resources failed to download or verify. ---')
    sys.exit(1)
else:
    print('--- [BUILD STEP] All specified NLTK resources successfully downloaded and verified during build. ---')