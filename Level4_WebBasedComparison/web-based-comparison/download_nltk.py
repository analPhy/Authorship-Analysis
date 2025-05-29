# download_nltk.py
import nltk
import sys
import os

nltk_data_dir = '/opt/render/nltk_data'

# Render環境でNLTK_DATA環境変数を設定し、nltk.data.pathにも追加
# (Renderの環境変数設定と合わせて二重で設定する形になるが、確実性を期す)
os.environ['NLTK_DATA'] = nltk_data_dir
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir) # 検索パスの先頭に追加

print(f'--- [BUILD STEP] Starting NLTK Resource Download to {nltk_data_dir} ---')
print(f'--- [BUILD STEP] NLTK search paths: {nltk.data.path} ---')

# アプリケーションが必要とするNLTKリソースのリスト
# (キー: nltk.download()で使うID, 値: nltk.data.find()で使う検索パス)
resources_to_download = {
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng.zip', # APIで実際に必要
    'punkt': 'tokenizers/punkt.zip',
    'words': 'corpora/words.zip',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker.zip',
    'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab.zip',
    # 'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger.zip' # 必要ならこれも
}

all_downloads_successful = True
for res_id, res_check_path_suffix in resources_to_download.items():
    # nltk.data.find は .zip を除いたディレクトリ名やファイル名を期待することが多い
    # しかし、ダウンロードIDとチェック用パスを調整する必要があるかもしれないので注意
    # ここでは、ダウンロードIDに対応する典型的な展開後のパスの一部を想定
    if res_id == 'averaged_perceptron_tagger_eng':
        check_path = 'taggers/averaged_perceptron_tagger_eng'
    elif res_id == 'punkt':
        check_path = 'tokenizers/punkt'
    elif res_id == 'words':
        check_path = 'corpora/words'
    elif res_id == 'maxent_ne_chunker':
        check_path = 'chunkers/maxent_ne_chunker'
    elif res_id == 'maxent_ne_chunker_tab':
        check_path = 'chunkers/maxent_ne_chunker_tab'
    else:
        check_path = res_check_path_suffix # その他の場合はIDをそのままパスの一部として使う (要調整)

    print(f'--- [BUILD STEP] Processing NLTK resource: {res_id} (expected at {check_path}) ---')
    try:
        nltk.data.find(check_path)
        print(f'--- [BUILD STEP] Resource {res_id} already exists. Skipping download. ---')
    except LookupError:
        print(f'--- [BUILD STEP] Resource {res_id} NOT FOUND. Attempting download... ---')
        try:
            if nltk.download(res_id, download_dir=nltk_data_dir, quiet=False, raise_on_error=True):
                print(f'--- [BUILD STEP] Successfully downloaded {res_id}. Verifying... ---')
                nltk.data.find(check_path) # ダウンロード後に再度検証
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
    sys.exit(1) # ひとつでも失敗したらビルドを失敗させる
else:
    print('--- [BUILD STEP] All specified NLTK resources successfully downloaded and verified during build. ---')