import nltk
import sys
import os

nltk_data_dir = '/opt/render/nltk_data'
resource_id_to_test = 'averaged_perceptron_tagger_eng'
resource_path_to_check = 'taggers/averaged_perceptron_tagger_eng' # nltk.data.find() で使うパス

os.environ['NLTK_DATA'] = nltk_data_dir
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

print(f'--- NLTK Test Download for: {resource_id_to_test} to {nltk_data_dir} ---')
print(f'NLTK search paths: {nltk.data.path}')

download_successful = False
try:
    print(f'Attempting to download: {resource_id_to_test}')
    if nltk.download(resource_id_to_test, download_dir=nltk_data_dir, quiet=False, raise_on_error=True):
        print(f'nltk.download for {resource_id_to_test} returned True.')
        download_successful = True
    else:
        # raise_on_error=True の場合、通常ここには到達しない
        print(f'nltk.download for {resource_id_to_test} returned False.')
except Exception as e:
    print(f'EXCEPTION during nltk.download for {resource_id_to_test}: {e}')

if download_successful:
    print(f'Download attempt for {resource_id_to_test} seems successful. Verifying presence...')
    try:
        found_path = nltk.data.find(resource_path_to_check)
        print(f'VERIFIED: {resource_id_to_test} found at {found_path}.')
        print('Build command considers this NLTK resource download successful.')
        sys.exit(0) # 明示的に成功で終了
    except LookupError:
        print(f'CRITICAL ERROR: {resource_id_to_test} NOT FOUND at {resource_path_to_check} after download attempt.')
        sys.exit(1) # 検証失敗ならビルドエラー
else:
    print(f'CRITICAL ERROR: Download failed for {resource_id_to_test}.')
    sys.exit(1) # ダウンロード失敗ならビルドエラー