// src/App.tsx
import React, { useState } from 'react';
import axios from 'axios';

// Flaskからの検索結果の型定義
interface SearchResult {
    context_words: string[]; // 前後の単語と一致した単語を含む単語リスト
    matched_start: number; // context_words内で一致した単語が始まるインデックス
    matched_end: number; // context_words内で一致した単語が終わる次のインデックス
}

const App: React.FC = () => {
  const [url, setUrl] = useState('https://en.wikipedia.org/wiki/Banana');
  const [phrase, setPhrase] = useState("");
  // 結果の型をSearch Result[]に変更
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState("");
  const [searchAttempted, setSearchAttempted] = useState(false);
  const [loading, setLoading] = useState(false);

  // 入力フィールド変更時に関連ステートをクリアする共通関数
  const handleInputChange = () => {
      setError("");
      setResults([]);
      setSearchAttempted(false);
      // loadingステートは検索ボタンクリック時のみ制御
  };

  const handleSearch = async () => {
    // 検索前のクリアとフラグ設定
    setError("");
    setResults([]);
    setSearchAttempted(true);
    setLoading(true);

    // クライアント側での基本的な入力検証
    if (!url.trim()) {
        setError("Please provide a Wikipedia URL.");
        setSearchAttempted(false);
        setLoading(false);
        return;
    }
     if (!phrase.trim()) {
        setError("Please provide a phrase to search.");
        setSearchAttempted(false);
        setLoading(false);
        return;
    }
    // 単語数の検証もクライアント側で追加しておくとサーバー負荷を減らせる
    if (phrase.trim().split(/\s+/).length > 2) { // \s+ で複数の空白を区切り文字とする
        setError("Please enter one or two words only.");
        setSearchAttempted(false);
        setLoading(false);
        return;
    }


    try {
      const response = await axios.post(
        "http://localhost:5000/api/search",
        { url: url.trim(), phrase: phrase.trim() },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      // Flaskから返されたレスポンスボディを確認 (エラーは200 OKに含まれる場合あり)
      if (response.data.error) {
        setError(response.data.error);
        setResults([]); // エラー時は結果をクリア
      } else {
        // 成功時は結果を設定 (型はSearch Result[]を期待)
        setResults(response.data.results);
        setError(""); // 成功時はエラーをクリア
      }

    } catch (err: any) {
      // ネットワークエラーやFlaskが返す500などのエラー（400はFlask側でcatchして200や400+bodyで返すようにしたのでここには来にくい）
      console.error("Search API error:", err); // デバッグ用にエラー内容をコンソール出力
      // エラーレスポンスがあればそのエラーメッセージを使用、なければ一般的なメッセージ
      // AxiosErrorの構造を確認し、より詳細な情報を取得できる場合は修正
      const errorMessage = err.response?.data?.error || err.message || "An unknown error occurred.";
      setError(`Search failed: ${errorMessage}`);
      setResults([]);
    } finally {
        // 検索処理の最後にローディング終了
        setLoading(false);
    }
  };

  return (
    <div>
      <h1>Wikipedia Phrase Search</h1>

      <div>
          <label htmlFor="wiki-url">Wikipedia URL:</label>
          <input
            id="wiki-url"
            type="text"
            value={url}
            onChange={(e) => {
                setUrl(e.target.value);
                handleInputChange();
            }}
            placeholder="Enter Wikipedia URL"
            style={{ width: '80%', marginRight: '10px', padding: '5px' }}
          />
      </div>

      <div style={{ marginTop: '10px' }}>
           <label htmlFor="search-phrase">Phrase:</label>
            <input
                id="search-phrase"
                type="text"
                value={phrase}
                onChange={(e) => {
                    setPhrase(e.target.value);
                    handleInputChange();
                }}
                placeholder="Enter a word or phrase (max 2 words)"
                style={{ marginRight: '10px', padding: '5px' }}
            />
            {/* ローディング中はボタンを無効化＆テキスト変更、入力が不完全でも無効化 */}
            <button onClick={handleSearch} disabled={loading || !url.trim() || !phrase.trim() || phrase.trim().split(/\s+/).length > 2}>
                {loading ? 'Loading...' : 'Search'}
            </button>
      </div>

      {/* エラーメッセージはerrorステートに基づいて表示 */}
      {error && <p style={{ color: "red", marginTop: '10px' }}>{error}</p>}

      {/* 結果がある場合のみリストを表示 */}
      {results.length > 0 && (
        <div style={{ marginTop: '20px' }}>
            <h2>Search Results ({results.length} matches)</h2>
            <ul>
                {/* ★ここを修正：dangerouslySetInnerHTML を廃止★ */}
                {results.map((result, index) => (
                    <li key={index} style={{ marginBottom: '15px', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
                        {/* コンテキスト単語リストをループ処理 */}
                        {result.context_words.map((word, wordIndex) => (
                            <React.Fragment key={wordIndex}>
                                {/* 一致した範囲の単語だけを強調 */}
                                {wordIndex >= result.matched_start && wordIndex < result.matched_end ? (
                                    <strong style={{ backgroundColor: 'yellow' }}>{word}</strong> // 見つけやすく黄色背景に
                                ) : (
                                    word
                                )}
                                {/* 最後の単語以外はスペースを追加 */}
                                {wordIndex < result.context_words.length - 1 && ' '}
                            </React.Fragment>
                        ))}
                    </li>
                ))}
            </ul>
        </div>
      )}

      {/* 結果がなくてエラーもなく、かつ検索が一度でも試みられた場合に「見つかりませんでした」を表示 */}
      {!error && results.length === 0 && searchAttempted && !loading && (
         <p style={{ marginTop: '10px' }}>No results found for "{phrase}". Please check the URL and phrase.</p>
      )}

      {/* 初期表示時や入力中の何も表示されない状態 */}
       {!error && results.length === 0 && !searchAttempted && !loading && (
          <p style={{ marginTop: '10px', color: '#888' }}>Enter a Wikipedia URL and a phrase to search.</p>
       )}
    </div>
  );
};

export default App;