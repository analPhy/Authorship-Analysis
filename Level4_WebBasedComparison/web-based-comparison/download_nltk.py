import nltk
import sys
import os

nltk_data_dir = '/opt/render/nltk_data'
os.environ['NLTK_DATA'] = nltk_data_dir # NLTKがどこを探すべきか確実に伝える
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

print(f'--- NLTKビルドステップ: NLTKデータパス確認: {nltk.data.path} ---')

# ダウンロードと検証を行うNLTKリソースの辞書 (キー: download ID, 値: findで使うパス)
resources_to_verify = {
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'punkt': 'tokenizers/punkt',
    'words': 'corpora/words',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
    'maxent_ne_chunker_tab': 'chunkers/maxent_ne_chunker_tab'
}

all_resources_ok = True

for res_id, res_check_path in resources_to_verify.items():
    try:
        print(f'NLTKリソース確認: {res_id} (期待されるパス: {res_check_path})')
        nltk.data.find(res_check_path)
        print(f'リソース {res_id} は既に存在します。')
    except LookupError:
        print(f'リソース {res_id} が見つかりません。ダウンロードを試みます...')
        try:
            if nltk.download(res_id, download_dir=nltk_data_dir, quiet=False, raise_on_error=True): # quiet=Falseで詳細ログ、raise_on_error=Trueでエラー時に例外発生
                print(f'{res_id} のダウンロードに成功しました。検証中...')
                nltk.data.find(res_check_path) # ダウンロード後に再度検証
                print(f'リソース {res_id} のダウンロード後の検証に成功しました。')
            else:
                # raise_on_error=True のため、通常ここには到達しないはず
                print(f'エラー: nltk.download({res_id}) がFalseを返しました。')
                all_resources_ok = False
        except Exception as e:
            print(f'エラー: {res_id} のダウンロードまたは検証中に例外が発生しました: {e}')
            all_resources_ok = False

if not all_resources_ok:
    print('致命的なビルドエラー: 1つ以上のNLTKリソースのダウンロードまたは検証に失敗しました。')
    sys.exit(1) # ビルドを失敗させる
else:
    print('指定されたすべてのNLTKリソースのダウンロードと検証にビルド中に成功しました。')