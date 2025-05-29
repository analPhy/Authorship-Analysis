// src/App.tsx

// --- Imports ---
// EN: Import React and axios for component and HTTP requests
// JP: Reactとaxiosをインポート（コンポーネント作成とHTTPリクエスト用）
import React, { useState /* useMemo was not used, can be removed if not planned elsewhere */ } from 'react';
import axios from 'axios';

// --- KWIC Search Types (旧 Phrase Search) ---
// EN: Types for KWIC search results and API response
// JP: KWIC検索結果とAPIレスポンス用の型定義
interface KWICSearchResult {
    context_words: string[]; // EN: Words around the match / JP: マッチ周辺の単語配列
    matched_start: number;   // EN: Start index of match / JP: マッチ開始インデックス
    matched_end: number;     // EN: End index of match / JP: マッチ終了インデックス
}

interface KWICSearchResponse {
    results: KWICSearchResult[]; // EN: Array of KWIC results / JP: KWIC結果の配列
    error?: string;              // EN: Optional error message / JP: エラーメッセージ（任意）
}

// --- Authorship Analysis Types (変更なし) ---
// EN: Types for authorship analysis results
// JP: 著者識別分析結果用の型定義
interface DistinctiveWords {
    AuthorA: string[]; // EN: Distinctive words for Author A / JP: 著者Aの特徴語
    AuthorB: string[]; // EN: Distinctive words for Author B / JP: 著者Bの特徴語
    [key: string]: string[]; // EN: For extensibility / JP: 拡張用
}

interface SamplePrediction {
    sentence_snippet: string; // EN: Sentence sample / JP: 文の抜粋
    true_label: string;       // EN: True author label / JP: 正解ラベル
    predicted_label: string;  // EN: Predicted author label / JP: 予測ラベル
}

interface AuthorshipAnalysisResult {
    accuracy: string;                       // EN: Model accuracy / JP: モデル精度
    classification_report: string;          // EN: Classification report / JP: 分類レポート
    distinctive_words: DistinctiveWords;    // EN: Distinctive words / JP: 特徴語
    sample_predictions: SamplePrediction[]; // EN: Sample predictions / JP: サンプル予測
    training_samples_count: number;         // EN: Training sample count / JP: 訓練サンプル数
    test_samples_count: number;             // EN: Test sample count / JP: テストサンプル数
    error?: string;                         // EN: Optional error / JP: エラー（任意）
}

// EN: Search type options for KWIC
// JP: KWIC検索タイプの選択肢
type SearchType = 'token' | 'pos' | 'entity';

const App: React.FC = () => {
  // --- State Hooks ---
  // EN: State for KWIC search
  // JP: KWIC検索用の状態管理
  const [url, setUrl] = useState('https://en.wikipedia.org/wiki/Banana'); // Placeholder will be more generic below
  const [searchQuery, setSearchQuery] = useState("");
  const [searchType, setSearchType] = useState<SearchType>('token');
  const [kwicResults, setKwicResults] = useState<KWICSearchResult[]>([]);
  const [kwicError, setKwicError] = useState("");
  const [kwicSearchAttempted, setKwicSearchAttempted] = useState(false);
  const [kwicLoading, setKwicLoading] = useState(false);

  // EN: State for KWIC display settings
  // JP: KWIC表示設定用の状態管理
  const [highlightBgColor, setHighlightBgColor] = useState<string>('#f1c40f');
  const [highlightTextColor, setHighlightTextColor] = useState<string>('#2c3e50');
  const [contextWindowSize, setContextWindowSize] = useState<number>(5);

  // EN: State for authorship analysis
  // JP: 著者識別分析用の状態管理
  const [urlA, setUrlA] = useState('https://en.wikipedia.org/wiki/Plato'); // Placeholder will be more generic below
  const [urlB, setUrlB] = useState('https://en.wikipedia.org/wiki/Aristotle'); // Placeholder will be more generic below
  const [authorshipResult, setAuthorshipResult] = useState<AuthorshipAnalysisResult | null>(null);
  const [authorshipError, setAuthorshipError] = useState("");
  const [authorshipLoading, setAuthorshipLoading] = useState(false);

  // --- Input Handlers ---
  const handleKwicInputChange = () => {
      setKwicError("");
      setKwicResults([]);
      setKwicSearchAttempted(false);
  };
  
  const handleAuthorshipInputChange = () => {
    setAuthorshipError("");
    setAuthorshipResult(null);
  };

  // --- KWIC Search Handler ---
  const handleKwicSearch = async () => {
    setKwicError("");
    setKwicResults([]);
    setKwicSearchAttempted(true);
    setKwicLoading(true);

    if (!url.trim()) {
        setKwicError("Please provide a Web Page URL."); // Changed
        setKwicSearchAttempted(false); setKwicLoading(false); return;
    }
    if (!searchQuery.trim()) {
        setKwicError("Please provide a search query (token, POS tag, or entity).");
        setKwicSearchAttempted(false); setKwicLoading(false); return;
    }
    if (searchType === 'token' && searchQuery.trim().split(/\s+/).length > 5) { 
        setKwicError("For token search, please enter one to five words only.");
        setKwicSearchAttempted(false); setKwicLoading(false); return;
    }
    if ((searchType === 'pos' || searchType === 'entity') && searchQuery.trim().includes(" ")) {
      setKwicError(`For ${searchType} search, please enter a single tag/entity type (e.g., NNP, PERSON). No spaces allowed.`);
      setKwicSearchAttempted(false); setKwicLoading(false); return;
    }

    try {
      const response = await axios.post<KWICSearchResponse>(
        "http://localhost:8080/api/search", // Ensure this port matches your Flask server
        { 
            url: url.trim(), 
            query: searchQuery.trim(),
            type: searchType
        },
        { headers: { "Content-Type": "application/json" } }
      );
      if (response.data.error) {
        setKwicError(response.data.error);
        setKwicResults([]);
      } else {
        // POSタグ検索の場合はresultsを平坦化
        if (searchType === "pos" && Array.isArray(response.data.results)) {
          const flatResults: KWICSearchResult[] = [];
          response.data.results.forEach((group: any) => {
            if (Array.isArray(group.contexts)) {
              group.contexts.forEach((ctx: any) => {
                // context_words等が存在する場合のみ追加
                if (ctx && Array.isArray(ctx.context_words)) {
                  flatResults.push({
                    context_words: ctx.context_words,
                    matched_start: ctx.matched_start,
                    matched_end: ctx.matched_end,
                  });
                }
              });
            }
          });
          setKwicResults(flatResults);
        } else {
          setKwicResults(response.data.results || []);
        }
        setKwicError("");
      }
    } catch (err: any) {
      console.error("KWIC Search API error:", err);
      const errorMessage = err.response?.data?.error || err.message || "An unknown error occurred.";
      setKwicError(`Search failed: ${errorMessage}`);
      setKwicResults([]);
    } finally {
        setKwicLoading(false);
    }
  };

  // --- Authorship Analysis Handler ---
  const handleAuthorshipAnalysis = async () => {
    setAuthorshipError("");
    setAuthorshipResult(null);
    setAuthorshipLoading(true);

    if (!urlA.trim() || !urlB.trim()) {
        setAuthorshipError("Please provide two Web Page URLs for authorship analysis."); // Changed
        setAuthorshipLoading(false); return;
    }

    const isValidUrl = (urlString: string): boolean => {
        try {
            const parsedUrl = new URL(urlString);
            return parsedUrl.protocol === "http:" || parsedUrl.protocol === "https:";
        } catch (e) {
            return false;
        }
    };

    if (!isValidUrl(urlA.trim())) {
        setAuthorshipError("URL for Source A is invalid. Please enter a valid URL (e.g., https://example.com/article1)."); // Changed
        setAuthorshipLoading(false); return;
    }
    if (!isValidUrl(urlB.trim())) {
        setAuthorshipError("URL for Source B is invalid. Please enter a valid URL (e.g., https://example.com/article2)."); // Changed
        setAuthorshipLoading(false); return;
    }

    try {
        const response = await axios.post<AuthorshipAnalysisResult>(
            "http://localhost:8080/api/authorship", // Ensure this port matches your Flask server
            { url_a: urlA.trim(), url_b: urlB.trim() },
            { headers: { "Content-Type": "application/json" } }
        );
        if (response.data.error) {
            setAuthorshipError(response.data.error);
            setAuthorshipResult(null);
        } else {
            setAuthorshipResult(response.data);
            setAuthorshipError("");
        }
    } catch (err: any) {
        console.error("Authorship API error:", err);
        const errorMessage = err.response?.data?.error || err.message || "An unknown error occurred during authorship analysis.";
        setAuthorshipError(`Authorship analysis failed: ${errorMessage}`);
        setAuthorshipResult(null);
    } finally {
        setAuthorshipLoading(false);
    }
  };
  
  // --- KWIC Context Display Helper ---
  const getDisplayedContext = (result: KWICSearchResult) => {
    const { context_words, matched_start, matched_end } = result;
    const displayStart = Math.max(0, matched_start - contextWindowSize);
    const displayEnd = Math.min(context_words.length, matched_end + contextWindowSize);
    
    const wordsToDisplay: Array<{type: 'word' | 'ellipsis', content: string, isMatched?: boolean}> = [];

    if (displayStart > 0) {
        wordsToDisplay.push({ type: 'ellipsis', content: '... ' });
    }

    for (let i = displayStart; i < displayEnd; i++) {
        wordsToDisplay.push({
            type: 'word',
            content: context_words[i],
            isMatched: i >= matched_start && i < matched_end
        });
    }

    if (displayEnd < context_words.length) {
        wordsToDisplay.push({ type: 'ellipsis', content: ' ...' });
    }
    return wordsToDisplay;
  };

  // --- Main Render ---
  return (
    <div style={{ fontFamily: 'Arial, sans-serif', maxWidth: '900px', margin: '20px auto', padding: '20px', color: '#333' }}>
      <header style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1 style={{ fontSize: '2.5em', color: '#2c3e50' }}>Web Page Text Tools</h1> {/* Changed Title */}
      </header>

      <section style={{ marginBottom: '40px', padding: '25px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <h2 style={{ marginTop: 0, borderBottom: '2px solid #3498db', paddingBottom: '10px', color: '#3498db' }}>KWIC Search (Key Word In Context)</h2>
        <p style={{color: "#666", fontSize: "0.9em", marginBottom: '15px'}}>
            Enter a public web page URL to search for keywords, POS tags, or entities within its main textual content. 
            Results may vary based on website structure.
        </p>
        
        <div style={{ marginBottom: '15px' }}>
            <label htmlFor="kwic-url" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Web Page URL:</label> {/* Changed */}
            <input
              id="kwic-url"
              type="text"
              value={url}
              onChange={(e) => { setUrl(e.target.value); handleKwicInputChange(); }}
              placeholder="e.g., https://example.com/some-article-page" // Changed
              style={{ width: 'calc(100% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
        </div>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '15px', flexWrap: 'wrap' }}>
            <div style={{ flex: '1 1 200px', minWidth: '180px' }}>
                <label htmlFor="search-type" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Search Type:</label>
                <select 
                    id="search-type"
                    value={searchType}
                    onChange={(e) => { setSearchType(e.target.value as SearchType); setSearchQuery(""); handleKwicInputChange(); }}
                    style={{ width: '100%', padding: '10px', border: '1px solid #ccc', borderRadius: '4px', height: '42px', boxSizing: 'border-box' }}
                >
                    <option value="token">Token(s)</option>
                    <option value="pos">POS Tag</option>
                    <option value="entity">Entity Type</option>
                </select>
            </div>
            <div style={{ flex: '2 1 300px', minWidth: '250px' }}>
                <label htmlFor="search-query" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                    {searchType === 'token' ? 'Keyword(s):' : searchType === 'pos' ? 'POS Tag (e.g., NNP, VBZ):' : 'Entity Type (e.g., PERSON, ORG):'}
                </label>
                <input
                    id="search-query"
                    type="text"
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); handleKwicInputChange(); }}
                    placeholder={
                        searchType === 'token' ? "e.g., important concept (max 5 words)" : // Changed
                        searchType === 'pos' ? "e.g., VBZ" :
                        "e.g., GPE (Geo-Political Entity)"
                    }
                    style={{ width: 'calc(100% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px', boxSizing: 'border-box' }}
                />
            </div>
        </div>
        
        <div style={{ marginBottom: '20px' }}>
            <button 
                onClick={handleKwicSearch} 
                disabled={kwicLoading || !url.trim() || !searchQuery.trim()}
                style={{ padding: '10px 20px', background: '#3498db', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '1em', opacity: (kwicLoading || !url.trim() || !searchQuery.trim()) ? 0.6 : 1 }}
            >
                {kwicLoading ? 'Searching...' : 'Search KWIC'}
            </button>
        </div>

        <details style={{ marginBottom: '20px', border: '1px solid #eee', padding: '12px', borderRadius: '4px', background: '#fdfdfd' }}>
            <summary style={{ fontWeight: 'bold', cursor: 'pointer', color: '#2980b9', userSelect: 'none' }}>Display Settings</summary>
            <div style={{ marginTop: '15px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                <div>
                    <label htmlFor="highlight-bg-color" style={{ display: 'block', marginBottom: '5px', fontSize: '0.9em' }}>Highlight Background:</label>
                    <input type="color" id="highlight-bg-color" value={highlightBgColor} onChange={(e) => setHighlightBgColor(e.target.value)} style={{height: '35px', width: '70px', border: '1px solid #ccc', borderRadius: '3px', cursor: 'pointer'}}/>
                </div>
                <div>
                    <label htmlFor="highlight-text-color" style={{ display: 'block', marginBottom: '5px', fontSize: '0.9em' }}>Highlight Text Color:</label>
                    <input type="color" id="highlight-text-color" value={highlightTextColor} onChange={(e) => setHighlightTextColor(e.target.value)} style={{height: '35px', width: '70px', border: '1px solid #ccc', borderRadius: '3px', cursor: 'pointer'}}/>
                </div>
                <div>
                    <label htmlFor="context-window" style={{ display: 'block', marginBottom: '5px', fontSize: '0.9em' }}>Context Window (words before/after):</label>
                    <input 
                        type="number" 
                        id="context-window" 
                        value={contextWindowSize} 
                        onChange={(e) => setContextWindowSize(Math.max(0, parseInt(e.target.value, 10) || 0))}
                        min="0" 
                        max="20"
                        style={{ padding: '8px', width: '70px', border: '1px solid #ccc', borderRadius: '4px', boxSizing: 'border-box' }} 
                    />
                </div>
            </div>
        </details>
        
        {kwicError && <p style={{ color: "#e74c3c", marginTop: '15px', background: '#fceded', padding: '10px', borderRadius: '4px' }}>⚠️ {kwicError}</p>}
        
        {kwicResults.length > 0 && !kwicLoading && (
          <div style={{ marginTop: '25px' }}>
              <h3 style={{ color: '#2980b9' }}>Search Results <span style={{fontSize: '0.8em', color: '#7f8c8d'}}>({kwicResults.length} matches)</span></h3>
              <ul style={{ listStyleType: 'none', padding: 0 }}>
                  {kwicResults.map((result, index) => {
                      const displayedItems = getDisplayedContext(result);
                      return (
                        <li 
                            key={index} 
                            style={{ 
                                marginBottom: '18px', 
                                borderBottom: '1px dashed #eee', 
                                paddingBottom: '18px', 
                                lineHeight: '1.7', 
                                textAlign: 'left',
                                display: 'flex',
                                alignItems: 'flex-start'
                            }}
                        >
                            <span style={{ marginRight: '10px', fontWeight: 'normal', color: '#555', minWidth: '25px', textAlign: 'right' }}>
                                {index + 1}.
                            </span>
                            <div>
                                {displayedItems.map((item, itemIndex) => (
                                    <React.Fragment key={itemIndex}>
                                        {item.type === 'word' ? (
                                            item.isMatched ? (
                                                <strong style={{ backgroundColor: highlightBgColor, color: highlightTextColor, padding: '2px 4px', borderRadius: '3px' }}>{item.content}</strong>
                                            ) : (
                                                item.content
                                            )
                                        ) : (
                                            <span style={{color: '#7f8c8d', fontStyle: 'italic'}}>{item.content}</span>
                                        )}
                                        {(item.type === 'word' && itemIndex < displayedItems.length -1 && displayedItems[itemIndex+1].type === 'word') ? ' ' : ''}
                                    </React.Fragment>
                                ))}
                            </div>
                        </li>
                      );
                  })}
              </ul>
          </div>
        )}
        {!kwicError && kwicResults.length === 0 && kwicSearchAttempted && !kwicLoading && (
           <p style={{ marginTop: '15px', color: '#7f8c8d', background: '#ecf0f1', padding: '10px', borderRadius: '4px' }}>No results found for "{searchQuery}" (Type: {searchType}).</p>
        )}
        {!kwicError && kwicResults.length === 0 && !kwicSearchAttempted && !kwicLoading && (
           <p style={{ marginTop: '15px', color: '#95a5a6' }}>Enter a Web Page URL and query to begin KWIC searching.</p> // Changed
        )}
      </section>

      <section style={{ padding: '25px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <h2 style={{ marginTop: 0, borderBottom: '2px solid #27ae60', paddingBottom: '10px', color: '#27ae60' }}>Authorship Analysis</h2>
        <p style={{color: "#555", fontSize: "0.95em", marginBottom: '20px'}}>
            Compare writing styles from two public web page articles. 
            Analysis is more meaningful for pages with substantial, authored text.
        </p>
        
        <div style={{ marginBottom: '15px' }}>
            <label htmlFor="author-a-url" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>URL for Source A:</label> {/* Changed */}
            <input
                id="author-a-url"
                type="text"
                value={urlA}
                onChange={(e) => { setUrlA(e.target.value); handleAuthorshipInputChange(); }}
                placeholder="e.g., https://example.com/article-by-author-A" // Changed
                style={{ width: 'calc(100% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
        </div>
        <div style={{ marginBottom: '20px' }}>
            <label htmlFor="author-b-url" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>URL for Source B:</label> {/* Changed */}
            <input
                id="author-b-url"
                type="text"
                value={urlB}
                onChange={(e) => { setUrlB(e.target.value); handleAuthorshipInputChange(); }}
                placeholder="e.g., https://example.com/another-article" // Changed
                style={{ width: 'calc(100% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
        </div>
        <button 
            onClick={handleAuthorshipAnalysis} 
            disabled={authorshipLoading || !urlA.trim() || !urlB.trim()} 
            style={{ padding: '12px 25px', background: '#27ae60', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '1.05em', opacity: (authorshipLoading || !urlA.trim() || !urlB.trim()) ? 0.6 : 1 }}
        >
            {authorshipLoading ? 'Analyzing...' : 'Analyze Authorship'}
        </button>

        {authorshipError && <p style={{ color: "#e74c3c", marginTop: '15px', background: '#fceded', padding: '10px', borderRadius: '4px' }}>⚠️ {authorshipError}</p>}

        {authorshipResult && !authorshipLoading && (
            <div style={{ marginTop: '25px' }}>
                <h3 style={{color: '#16a085'}}>Analysis Results</h3>
                <div style={{ background: '#ecf0f1', padding: '15px', borderRadius: '4px', marginBottom: '20px' }}>
                    <p style={{margin: '5px 0'}}><strong>Model Accuracy:</strong> <span style={{fontSize: '1.1em', color: '#2c3e50'}}>{authorshipResult.accuracy}</span></p>
                    <p style={{margin: '5px 0'}}><strong>Training Samples:</strong> {authorshipResult.training_samples_count}</p>
                    <p style={{margin: '5px 0'}}><strong>Test Samples:</strong> {authorshipResult.test_samples_count}</p>
                </div>
                
                <div style={{ marginBottom: '20px' }}>
                    <h4 style={{color: '#16a085'}}>Classification Report:</h4>
                    <pre style={{ background: '#e8f6f3', padding: '15px', borderRadius: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-all', border: '1px solid #d0e9e1', fontSize: '0.9em', lineHeight: '1.6' }}>
                        {authorshipResult.classification_report}
                    </pre>
                </div>

                <div style={{ marginBottom: '20px' }}>
                    <h4 style={{color: '#16a085'}}>Most Distinctive Words/N-grams (Top 10):</h4>
                    <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '15px' }}>
                        <div style={{flex: '1 1 45%', minWidth: '280px', marginBottom: '10px'}}>
                            <strong>Source A (from <a href={urlA} target="_blank" rel="noopener noreferrer" style={{color: '#2980b9'}}>{urlA.split('/').pop() || urlA}</a>):</strong> {/* Changed */}
                            <ul style={{ listStyleType: 'square', paddingLeft: '20px', background: '#fdfefe', border: '1px solid #eaeded', borderRadius: '4px', padding: '10px 10px 10px 30px', marginTop: '5px'}}>
                                {authorshipResult.distinctive_words.AuthorA?.map((word, i) => <li key={`a-${i}`} style={{marginBottom: '3px'}}>{word}</li>) || <li>N/A</li>}
                            </ul>
                        </div>
                        <div style={{flex: '1 1 45%', minWidth: '280px'}}>
                            <strong>Source B (from <a href={urlB} target="_blank" rel="noopener noreferrer" style={{color: '#2980b9'}}>{urlB.split('/').pop() || urlB}</a>):</strong> {/* Changed */}
                            <ul style={{ listStyleType: 'square', paddingLeft: '20px', background: '#fdfefe', border: '1px solid #eaeded', borderRadius: '4px', padding: '10px 10px 10px 30px', marginTop: '5px'}}>
                                {authorshipResult.distinctive_words.AuthorB?.map((word, i) => <li key={`b-${i}`} style={{marginBottom: '3px'}}>{word}</li>) || <li>N/A</li>}
                            </ul>
                        </div>
                    </div>
                </div>

                <div>
                    <h4 style={{color: '#16a085'}}>Sample Predictions on Test Sentences (Max 5):</h4>
                    <ul style={{ listStyleType: 'none', padding: 0 }}>
                        {authorshipResult.sample_predictions.map((pred, i) => (
                            <li key={`pred-${i}`} style={{ border: '1px solid #eaeded', background: '#fdfefe', padding: '12px', marginBottom: '10px', borderRadius: '4px' }}>
                                <p style={{ margin: '0 0 8px 0', fontStyle: 'italic', color: '#566573' }}>"{pred.sentence_snippet}"</p>
                                <p style={{ margin: 0, fontSize: '0.95em' }}>
                                    <span style={{ 
                                        fontWeight: 'bold', 
                                        color: pred.true_label === pred.predicted_label ? '#229954' : '#c0392b',
                                        padding: '3px 6px',
                                        borderRadius: '3px',
                                        background: pred.true_label === pred.predicted_label ? '#d4efdf' : '#f5b7b1'
                                    }}>
                                        Predicted: {pred.predicted_label} 
                                    </span>
                                    <span style={{marginLeft: '10px', color: '#7f8c8d'}}>(True: {pred.true_label})</span>
                                </p>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        )}
         {!authorshipError && !authorshipResult && !authorshipLoading && (
           <p style={{ marginTop: '15px', color: '#95a5a6' }}>Enter two Web Page URLs and click "Analyze Authorship" to view the analysis.</p> // Changed
        )}
      </section>
      <footer style={{textAlign: 'center', marginTop: '40px', paddingTop: '20px', borderTop: '1px solid #eee', fontSize: '0.9em', color: '#7f8c8d'}}>
        <p>Web Page Text Tools - KWIC & Authorship Analysis</p> {/* Changed */}
        <p style={{fontSize: '0.8em', color: '#95a5a6'}}>
            Please ensure you have the right to process content from the provided URLs. Results depend on website structure.
        </p>
      </footer>
    </div>
  );
};

export default App;