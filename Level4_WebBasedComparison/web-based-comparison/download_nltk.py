# download_nltk.py
import nltk
import sys
import os

nltk_data_dir = '/opt/render/nltk_data'
# NLTK_DATA環境変数を設定し、検索パスに明示的に追加する
# (Renderの環境変数で設定されていれば不要な場合もあるが、念のため)
os.environ['NLTK_DATA'] = nltk_data_dir
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

print(f'--- Starting NLTK Resource Download to {nltk_data_dir} ---')
print(f'NLTK search paths: {nltk.data.path}')

resources_to_download = [
    ('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng'),
    ('punkt', 'tokenizers/punkt'),
    ('words', 'corpora/words'),
    ('maxent_ne_chunker', 'chunkers/maxent_ne_chunker'),
    ('maxent_ne_chunker_tab', 'chunkers/maxent_ne_chunker_tab')
]

all_downloads_successful = True
for res_id, res_path_check in resources_to_download:
    print(f'Attempting to download NLTK resource: {res_id}')
    try:
        # まず存在確認
        try:
            nltk.data.find(res_path_check)
            print(f'Resource {res_id} already exists at {res_path_check}. Skipping download.')
            continue # 既に存在すればダウンロードしない
        except LookupError:
            pass # 存在しない場合はダウンロードへ

        if nltk.download(res_id, download_dir=nltk_data_dir, quiet=False, raise_on_error=True):
            print(f'Successfully downloaded NLTK resource: {res_id}')
            # ダウンロード後にもう一度存在確認
            try:
                nltk.data.find(res_path_check)
                print(f'Verified resource {res_id} at {res_path_check} after download.')
            except LookupError:
                print(f'ERROR: Resource {res_id} NOT FOUND at {res_path_check} even after attempting download.')
                all_downloads_successful = False
        else:
            # nltk.downloadがFalseを返した場合 (raise_on_error=Falseの時など)
            print(f'ERROR: nltk.download returned False for {res_id}. Download may have failed.')
            all_downloads_successful = False
    except Exception as e:
        print(f'ERROR: Exception occurred while downloading NLTK resource {res_id}: {e}')
        all_downloads_successful = False

print('--- NLTK Resource Download Finished ---')
if not all_downloads_successful:
    print('CRITICAL: One or more NLTK resources failed to download or verify. Failing build.')
    sys.exit(1)
else:
    print('All NLTK resources downloaded and verified successfully.')