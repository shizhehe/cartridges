import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [examples, setExamples] = useState([])
  const [selectedExample, setSelectedExample] = useState(null)
  const [showLogprobsModal, setShowLogprobsModal] = useState(false)
  const [selectedMessage, setSelectedMessage] = useState(null)
  const [outputDir, setOutputDir] = useState('')

  // Dataset discovery
  useEffect(() => {
    const discoverDatasets = async () => {
      try {
        const response = await fetch('/api/datasets')
        const data = await response.json()
        setDatasets(data)
      } catch (error) {
        console.error('Failed to discover datasets:', error)
      }
    }
    discoverDatasets()
  }, [outputDir])

  const loadDataset = async (datasetPath) => {
    try {
      const response = await fetch(`/api/dataset/${encodeURIComponent(datasetPath)}`)
      const data = await response.json()
      setExamples(data.examples)
      setSelectedDataset(datasetPath)
    } catch (error) {
      console.error('Failed to load dataset:', error)
    }
  }

  const openLogprobsModal = (message) => {
    setSelectedMessage(message)
    setShowLogprobsModal(true)
  }

  return (
    <div className="app">
      <div className="sidebar">
        <h2>Datasets</h2>
        <div className="output-dir-input">
          <input
            type="text"
            placeholder="Output directory path"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
          />
        </div>
        <div className="dataset-list">
          {datasets.map((dataset, idx) => (
            <div
              key={idx}
              className={`dataset-item ${selectedDataset === dataset.path ? 'selected' : ''}`}
              onClick={() => loadDataset(dataset.path)}
            >
              <div className="dataset-name">{dataset.name}</div>
              <div className="dataset-path">{dataset.path}</div>
              <div className="dataset-info">{dataset.size} examples</div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="main-content">
        {!selectedDataset ? (
          <div className="empty-state">
            <h2>Select a dataset to explore</h2>
            <p>Choose a dataset from the sidebar to begin exploring training examples.</p>
          </div>
        ) : !selectedExample ? (
          <div className="global-view">
            <h2>Examples Overview</h2>
            <div className="examples-grid">
              {examples.map((example, idx) => (
                <div
                  key={idx}
                  className="example-card"
                  onClick={() => setSelectedExample(example)}
                >
                  <div className="example-type">{example.type}</div>
                  <div className="example-messages">{example.messages.length} messages</div>
                  <div className="example-preview">
                    {example.messages[0]?.content.substring(0, 100)}...
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="focused-view">
            <div className="focused-header">
              <button onClick={() => setSelectedExample(null)}>← Back to Overview</button>
              <h2>Training Example - {selectedExample.type}</h2>
            </div>
            <div className="chat-view">
              {selectedExample.messages.map((message, idx) => (
                <div
                  key={idx}
                  className={`message ${message.role}`}
                  onClick={() => message.top_logprobs && openLogprobsModal(message)}
                >
                  <div className="message-role">{message.role}</div>
                  <div className="message-content">{message.content}</div>
                  {message.top_logprobs && (
                    <div className="logprobs-indicator">📊 Has logprobs</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      {showLogprobsModal && selectedMessage && (
        <LogprobsModal
          message={selectedMessage}
          onClose={() => setShowLogprobsModal(false)}
        />
      )}
    </div>
  )
}

function LogprobsModal({ message, onClose }) {
  const [tokensPerPage, setTokensPerPage] = useState(5)
  const [currentPage, setCurrentPage] = useState(0)

  if (!message.top_logprobs) return null

  const totalTokens = message.top_logprobs.logprobs.length
  const totalPages = Math.ceil(totalTokens / tokensPerPage)
  const startIdx = currentPage * tokensPerPage
  const endIdx = Math.min(startIdx + tokensPerPage, totalTokens)

  const decodeToken = (tokenId) => {
    return `[${tokenId}]`
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Token Logprobs</h3>
          <button onClick={onClose}>×</button>
        </div>
        
        <div className="logprobs-content">
          <div className="logprobs-controls">
            <div className="page-controls">
              <button 
                onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                disabled={currentPage === 0}
              >
                Previous
              </button>
              <span>Page {currentPage + 1} of {totalPages}</span>
              <button 
                onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                disabled={currentPage === totalPages - 1}
              >
                Next
              </button>
            </div>
            <div className="tokens-per-page">
              <label>
                Tokens per page:
                <select value={tokensPerPage} onChange={(e) => setTokensPerPage(Number(e.target.value))}>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                </select>
              </label>
            </div>
          </div>
          
          <div className="tokens-page">
            <h4>Tokens {startIdx + 1}-{endIdx} of {totalTokens}</h4>
            {Array.from({ length: endIdx - startIdx }, (_, i) => {
              const tokenIdx = startIdx + i
              const tokenLogprobs = message.top_logprobs.logprobs[tokenIdx]
              const tokenIds = message.top_logprobs.token_ids[tokenIdx]
              
              const probs = tokenLogprobs.map(logprob => Math.exp(logprob))
              const sortedIndices = probs.map((_, idx) => idx).sort((a, b) => probs[b] - probs[a])
              
              return (
                <div key={tokenIdx} className="token-details">
                  <div className="token-header">
                    <strong>Token {tokenIdx + 1}: {decodeToken(tokenIds[0])}</strong>
                  </div>
                  <div className="token-alternatives">
                    {sortedIndices.slice(0, 5).map((idx, rank) => (
                      <div key={idx} className={`alternative ${rank === 0 ? 'chosen' : ''}`}>
                        <span className="rank">{rank + 1}:</span>
                        <span className="token-text">{decodeToken(tokenIds[idx])}</span>
                        <span className="probability">{(probs[idx] * 100).toFixed(2)}%</span>
                        <span className="logprob">{tokenLogprobs[idx].toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
