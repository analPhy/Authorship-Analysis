// src/App.tsx
import React, { useState } from 'react';
import axios from 'axios';

// --- Phrase Search Types ---
interface SearchResult {
    context_words: string[];
    matched_start: number;
    matched_end: number;
}

interface PhraseSearchResponse {
    results: SearchResult[];
    error?: string;
}

// --- Authorship Analysis Types ---
interface DistinctiveWords {
    AuthorA: string[];
    AuthorB: string[];
    // Potentially other authors if the backend model changes
    [key: string]: string[]; // Index signature for dynamic author keys
}

interface SamplePrediction {
    sentence_snippet: string;
    true_label: string;
    predicted_label: string;
}

interface AuthorshipAnalysisResult {
    accuracy: string;
    classification_report: string;
    distinctive_words: DistinctiveWords;
    sample_predictions: SamplePrediction[];
    training_samples_count: number;
    test_samples_count: number;
    error?: string; // For errors returned in the JSON body with a 200/400 status
}


const App: React.FC = () => {
  // --- Phrase Search State ---
  const [url, setUrl] = useState('https://en.wikipedia.org/wiki/Banana');
  const [phrase, setPhrase] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState("");
  const [searchAttempted, setSearchAttempted] = useState(false);
  const [loading, setLoading] = useState(false);

  // --- Authorship Analysis State ---
  const [urlA, setUrlA] = useState('https://en.wikipedia.org/wiki/Plato');
  const [urlB, setUrlB] = useState('https://en.wikipedia.org/wiki/Aristotle');
  const [authorshipResult, setAuthorshipResult] = useState<AuthorshipAnalysisResult | null>(null);
  const [authorshipError, setAuthorshipError] = useState("");
  const [authorshipLoading, setAuthorshipLoading] = useState(false);


  const handleInputChange = () => {
      setError("");
      setResults([]);
      setSearchAttempted(false);
  };
  
  const handleAuthorshipInputChange = () => {
    setAuthorshipError("");
    setAuthorshipResult(null);
  };

  const handleSearch = async () => {
    setError("");
    setResults([]);
    setSearchAttempted(true);
    setLoading(true);

    if (!url.trim()) {
        setError("Please provide a Wikipedia URL.");
        setSearchAttempted(false); setLoading(false); return;
    }
    if (!phrase.trim()) {
        setError("Please provide a phrase to search.");
        setSearchAttempted(false); setLoading(false); return;
    }
    if (phrase.trim().split(/\s+/).length > 2) {
        setError("Please enter one or two words only.");
        setSearchAttempted(false); setLoading(false); return;
    }

    try {
      const response = await axios.post<PhraseSearchResponse>(
        "http://localhost:5000/api/search",
        { url: url.trim(), phrase: phrase.trim() },
        { headers: { "Content-Type": "application/json" } }
      );
      if (response.data.error) {
        setError(response.data.error);
        setResults([]);
      } else {
        setResults(response.data.results || []); // Ensure results is always an array
        setError("");
      }
    } catch (err: any) {
      console.error("Search API error:", err);
      const errorMessage = err.response?.data?.error || err.message || "An unknown error occurred.";
      setError(`Search failed: ${errorMessage}`);
      setResults([]);
    } finally {
        setLoading(false);
    }
  };

  const handleAuthorshipAnalysis = async () => {
    setAuthorshipError("");
    setAuthorshipResult(null);
    setAuthorshipLoading(true);

    if (!urlA.trim() || !urlB.trim()) {
        setAuthorshipError("Please provide two Wikipedia URLs for authorship analysis.");
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
        setAuthorshipError("URL for Author A is invalid. Please enter a valid URL (e.g., https://en.wikipedia.org/wiki/PageName).");
        setAuthorshipLoading(false); return;
    }
    if (!isValidUrl(urlB.trim())) {
        setAuthorshipError("URL for Author B is invalid. Please enter a valid URL (e.g., https://en.wikipedia.org/wiki/PageName).");
        setAuthorshipLoading(false); return;
    }

    try {
        const response = await axios.post<AuthorshipAnalysisResult>(
            "http://localhost:5000/api/authorship",
            { url_a: urlA.trim(), url_b: urlB.trim() },
            { headers: { "Content-Type": "application/json" } }
        );
        // Backend might send an error within a success (200) or client error (400) response body
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


  return (
    <div style={{ fontFamily: 'Arial, sans-serif', maxWidth: '900px', margin: '20px auto', padding: '20px', color: '#333' }}>
      <header style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h1 style={{ fontSize: '2.5em', color: '#2c3e50' }}>Wikipedia Text Tools</h1>
      </header>

      {/* --- Wikipedia Phrase Search Section --- */}
      <section style={{ marginBottom: '40px', padding: '25px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <h2 style={{ marginTop: 0, borderBottom: '2px solid #3498db', paddingBottom: '10px', color: '#3498db' }}>Phrase Search</h2>
        
        <div style={{ marginBottom: '15px' }}>
            <label htmlFor="wiki-url" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Wikipedia URL:</label>
            <input
              id="wiki-url"
              type="text"
              value={url}
              onChange={(e) => { setUrl(e.target.value); handleInputChange(); }}
              placeholder="e.g., https://en.wikipedia.org/wiki/Banana"
              style={{ width: 'calc(100% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
        </div>
        <div style={{ marginBottom: '15px' }}>
            <label htmlFor="search-phrase" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>Phrase (1-2 words):</label>
            <input
                id="search-phrase"
                type="text"
                value={phrase}
                onChange={(e) => { setPhrase(e.target.value); handleInputChange(); }}
                placeholder="e.g., yellow fruit"
                style={{ width: 'calc(70% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px', marginRight: '10px' }}
            />
            <button 
                onClick={handleSearch} 
                disabled={loading || !url.trim() || !phrase.trim() || phrase.trim().split(/\s+/).length > 2} 
                style={{ padding: '10px 20px', background: '#3498db', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '1em', opacity: (loading || !url.trim() || !phrase.trim() || phrase.trim().split(/\s+/).length > 2) ? 0.6 : 1 }}
            >
                {loading ? 'Searching...' : 'Search'}
            </button>
        </div>
        {error && <p style={{ color: "#e74c3c", marginTop: '15px', background: '#fceded', padding: '10px', borderRadius: '4px' }}>⚠️ {error}</p>}
        
        {results.length > 0 && !loading && (
          <div style={{ marginTop: '25px' }}>
              <h3 style={{ color: '#2980b9' }}>Search Results <span style={{fontSize: '0.8em', color: '#7f8c8d'}}>({results.length} matches)</span></h3>
              <ul style={{ listStyleType: 'none', padding: 0 }}>
                  {results.map((result, index) => (
                      <li key={index} style={{ marginBottom: '18px', borderBottom: '1px dashed #eee', paddingBottom: '18px', lineHeight: '1.7' }}>
                          {result.context_words.map((word, wordIndex) => (
                              <React.Fragment key={wordIndex}>
                                  {wordIndex >= result.matched_start && wordIndex < result.matched_end ? (
                                      <strong style={{ backgroundColor: '#f1c40f', padding: '2px 4px', borderRadius: '3px', color: '#2c3e50' }}>{word}</strong>
                                  ) : (
                                      word
                                  )}
                                  {wordIndex < result.context_words.length - 1 && ' '}
                              </React.Fragment>
                          ))}
                      </li>
                  ))}
              </ul>
          </div>
        )}
        {!error && results.length === 0 && searchAttempted && !loading && (
           <p style={{ marginTop: '15px', color: '#7f8c8d', background: '#ecf0f1', padding: '10px', borderRadius: '4px' }}>No results found for "{phrase}".</p>
        )}
        {!error && results.length === 0 && !searchAttempted && !loading && (
           <p style={{ marginTop: '15px', color: '#95a5a6' }}>Enter a Wikipedia URL and a phrase to begin searching.</p>
        )}
      </section>

      {/* --- Wikipedia Authorship Analysis Section --- */}
      <section style={{ padding: '25px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <h2 style={{ marginTop: 0, borderBottom: '2px solid #27ae60', paddingBottom: '10px', color: '#27ae60' }}>Authorship Analysis</h2>
        <p style={{color: "#555", fontSize: "0.95em", marginBottom: '20px'}}>Compare writing styles from two Wikipedia articles.</p>
        
        <div style={{ marginBottom: '15px' }}>
            <label htmlFor="author-a-url" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>URL for Author A:</label>
            <input
                id="author-a-url"
                type="text"
                value={urlA}
                onChange={(e) => { setUrlA(e.target.value); handleAuthorshipInputChange(); }}
                placeholder="e.g., https://en.wikipedia.org/wiki/Plato"
                style={{ width: 'calc(100% - 22px)', padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
        </div>
        <div style={{ marginBottom: '20px' }}>
            <label htmlFor="author-b-url" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>URL for Author B:</label>
            <input
                id="author-b-url"
                type="text"
                value={urlB}
                onChange={(e) => { setUrlB(e.target.value); handleAuthorshipInputChange(); }}
                placeholder="e.g., https://en.wikipedia.org/wiki/Aristotle"
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                        <div style={{flex: '1 1 45%', minWidth: '280px', marginBottom: '10px'}}>
                            <strong>Author A (from <a href={urlA} target="_blank" rel="noopener noreferrer" style={{color: '#2980b9'}}>{urlA.split('/').pop()}</a>):</strong>
                            <ul style={{ listStyleType: 'square', paddingLeft: '20px', background: '#fdfefe', border: '1px solid #eaeded', borderRadius: '4px', padding: '10px 10px 10px 30px', marginTop: '5px'}}>
                                {authorshipResult.distinctive_words.AuthorA?.map((word, i) => <li key={`a-${i}`} style={{marginBottom: '3px'}}>{word}</li>) || <li>N/A</li>}
                            </ul>
                        </div>
                        <div style={{flex: '1 1 45%', minWidth: '280px'}}>
                            <strong>Author B (from <a href={urlB} target="_blank" rel="noopener noreferrer" style={{color: '#2980b9'}}>{urlB.split('/').pop()}</a>):</strong>
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
           <p style={{ marginTop: '15px', color: '#95a5a6' }}>Enter two Wikipedia URLs and click "Analyze Authorship" to view the analysis.</p>
        )}
      </section>
      <footer style={{textAlign: 'center', marginTop: '40px', paddingTop: '20px', borderTop: '1px solid #eee', fontSize: '0.9em', color: '#7f8c8d'}}>
        <p>Wikipedia Text Tools</p>
      </footer>
    </div>
  );
};

export default App;