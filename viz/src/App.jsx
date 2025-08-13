import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

function App() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [examples, setExamples] = useState([])
  const [totalExamples, setTotalExamples] = useState(0)
  const [selectedExample, setSelectedExample] = useState(null)
  const [outputDir, setOutputDir] = useState('')
  const [tokenizerName, setTokenizerName] = useState('meta-llama/Llama-3.2-3B-Instruct')
  const [systemPromptExpanded, setSystemPromptExpanded] = useState(false)
  const [currentPage, setCurrentPage] = useState(0)
  const [examplesPerPage] = useState(128)
  const [loadingDatasetPath, setLoadingDatasetPath] = useState(null)

  // Dataset discovery
  useEffect(() => {
    const discoverDatasets = async () => {
      try {
        console.log('Fetching datasets...')
        const response = await fetch('/api/datasets')
        const data = await response.json()
        console.log('Datasets received:', data)
        // Ensure datasets are sorted by relative path (reverse alphabetical)
        const sortedData = data.sort((a, b) => b.relative_path.localeCompare(a.relative_path))
        setDatasets(sortedData)
      } catch (error) {
        console.error('Failed to discover datasets:', error)
      }
    }
    discoverDatasets()
  }, [outputDir])

  const loadDataset = async (datasetPath, page = 0) => {
    setLoadingDatasetPath(datasetPath)
    try {
      const response = await fetch(`/api/dataset/${encodeURIComponent(datasetPath)}?page=${page}&page_size=${examplesPerPage}`)
      const data = await response.json()
      setExamples(data.examples)
      setTotalExamples(data.total_count)
      setSelectedDataset(datasetPath)
      setCurrentPage(page)
    } catch (error) {
      console.error('Failed to load dataset:', error)
    } finally {
      setLoadingDatasetPath(null)
    }
  }


  const decodeTokens = async (tokenIds) => {
    try {
      const response = await fetch('/api/decode-tokens', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tokenizer_name: tokenizerName,
          token_ids: tokenIds
        })
      })
      const data = await response.json()
      return data.decoded_tokens || []
    } catch (error) {
      console.error('Failed to decode tokens:', error)
      return tokenIds.map(id => `[ID:${id}]`)
    }
  }

  const navigateExample = (direction) => {
    if (!selectedExample || examples.length === 0) return
    
    const currentIndex = examples.findIndex(ex => ex === selectedExample)
    let newIndex
    
    if (direction === 'next') {
      newIndex = (currentIndex + 1) % examples.length
    } else {
      newIndex = (currentIndex - 1 + examples.length) % examples.length
    }
    
    setSelectedExample(examples[newIndex])
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!selectedExample) return
      
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        navigateExample('next')
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        navigateExample('prev')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedExample, examples])

  // Token visualization component
  const TokenVisualization = ({ message }) => {
    const [decodedTokens, setDecodedTokens] = useState({})
    const [hoveredToken, setHoveredToken] = useState(null)
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0, showAbove: false })
    
    useEffect(() => {
      if (!message?.top_logprobs) return
      
      const decodeAllTokens = async () => {
        const allTokenIds = new Set()
        const numTokens = Math.min(100, message.top_logprobs.logprobs.length)
        
        for (let i = 0; i < numTokens; i++) {
          const tokenIds = message.top_logprobs.token_ids[i]
          tokenIds.forEach(id => allTokenIds.add(id))
        }
        
        const decoded = await decodeTokens(Array.from(allTokenIds))
        const decodedMap = {}
        Array.from(allTokenIds).forEach((id, idx) => {
          decodedMap[id] = decoded[idx] || `[ID:${id}]`
        })
        setDecodedTokens(decodedMap)
      }
      
      decodeAllTokens()
    }, [message, tokenizerName])
    
    if (!message?.top_logprobs) return null
    
    const numTokens = Math.min(100, message.top_logprobs.logprobs.length)
    const tokens = []
    
    for (let i = 0; i < numTokens; i++) {
      const tokenLogprobs = message.top_logprobs.logprobs[i]
      const tokenIds = message.top_logprobs.token_ids[i]
      
      const probs = tokenLogprobs.map(logprob => Math.exp(logprob))
      const sortedIndices = probs.map((_, idx) => idx).sort((a, b) => probs[b] - probs[a])
      
      const chosenText = decodedTokens[tokenIds[0]] || `[ID:${tokenIds[0]}]`
      
      // Get top alternatives
      const alternatives = sortedIndices.slice(0, 5).map((idx, rank) => ({
        text: decodedTokens[tokenIds[idx]] || `[ID:${tokenIds[idx]}]`,
        prob: probs[idx],
        logprob: tokenLogprobs[idx],
        rank: rank + 1,
        chosen: rank === 0
      }))
      
      tokens.push({
        text: chosenText,
        position: i,
        prob: probs[0],
        logprob: tokenLogprobs[0],
        alternatives,
        entropy: -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0)
      })
    }
    
    const getColorForProbability = (prob) => {
      // Clamp probability to [0, 1] and calculate color intensity
      const clampedProb = Math.max(0, Math.min(1, prob))
      
      // Use a continuous color scale from red (low) to yellow (medium) to green (high)
      let r, g, b
      
      if (clampedProb < 0.5) {
        // Red to Yellow transition (0 to 0.5)
        const t = clampedProb * 2 // Scale to [0, 1]
        r = 255
        g = Math.round(255 * t)
        b = 0
      } else {
        // Yellow to Green transition (0.5 to 1)
        const t = (clampedProb - 0.5) * 2 // Scale to [0, 1]
        r = Math.round(255 * (1 - t))
        g = 255
        b = 0
      }
      
      // Add transparency and return as inline style
      const backgroundColor = `rgba(${r}, ${g}, ${b}, 0.3)`
      const hoverBackgroundColor = `rgba(${r}, ${g}, ${b}, 0.5)`
      
      return {
        backgroundColor,
        hoverBackgroundColor
      }
    }
    
    const handleTokenHover = (event, token) => {
      const rect = event.currentTarget.getBoundingClientRect()
      const containerRect = event.currentTarget.closest('.token-container').getBoundingClientRect()
      const windowHeight = window.innerHeight
      
      // Calculate if tooltip should appear above or below
      const spaceBelow = windowHeight - rect.bottom
      const spaceAbove = rect.top
      const tooltipHeight = 200 // Approximate tooltip height
      
      const showAbove = spaceBelow < tooltipHeight && spaceAbove > tooltipHeight
      
      setTooltipPosition({
        x: rect.left + rect.width / 2,
        y: showAbove ? rect.top : rect.bottom,
        showAbove
      })
      setHoveredToken(token)
    }

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold mb-2 text-gray-800">Token Probabilities (hover for details)</h4>
        <div className="relative token-container">
          <div className="font-mono text-sm leading-relaxed">
            {tokens.map((token, idx) => {
              const colors = getColorForProbability(token.prob)
              return (
                <span
                  key={idx}
                  className="inline-block px-1 py-0.5 m-0.5 rounded cursor-pointer transition-all"
                  style={{
                    backgroundColor: colors.backgroundColor,
                    '--hover-bg': colors.hoverBackgroundColor
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = colors.hoverBackgroundColor
                    handleTokenHover(e, token)
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = colors.backgroundColor
                    setHoveredToken(null)
                  }}
                  title={`${token.text} (${(token.prob * 100).toFixed(1)}%)`}
                >
                  {token.text}
                </span>
              )
            })}
          </div>
        </div>
        
        {hoveredToken && (
          <div 
            className="fixed z-50 bg-white border border-gray-300 shadow-lg p-3 rounded-lg max-w-sm pointer-events-none" 
            style={{
              left: `${tooltipPosition.x}px`,
              top: tooltipPosition.showAbove 
                ? `${tooltipPosition.y - 8}px` 
                : `${tooltipPosition.y + 8}px`,
              transform: tooltipPosition.showAbove 
                ? 'translate(-50%, -100%)' 
                : 'translate(-50%, 0)'
            }}>
            <div className="font-semibold mb-2 text-gray-800">
              Token {hoveredToken.position + 1}: "{hoveredToken.text}"
            </div>
            <div className="text-sm space-y-1 text-gray-700">
              <div>Probability: {(hoveredToken.prob * 100).toFixed(2)}%</div>
              <div>Log Probability: {hoveredToken.logprob.toFixed(4)}</div>
              <div>Entropy: {hoveredToken.entropy.toFixed(3)}</div>
            </div>
            <div className="mt-2 pt-2 border-t border-gray-300">
              <div className="text-xs font-semibold mb-1 text-gray-800">Top Alternatives:</div>
              {hoveredToken.alternatives.map((alt, idx) => (
                <div key={idx} className={`text-xs ${alt.chosen ? 'text-green-700 font-semibold' : 'text-gray-600'}`}>
                  {alt.rank}. "{alt.text}" ({(alt.prob * 100).toFixed(1)}%)
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-screen font-sans">
      <div className="w-80 bg-gray-100 border-r border-gray-300 p-4 overflow-y-auto">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Datasets</h2>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Tokenizer
          </label>
          <select
            value={tokenizerName}
            onChange={(e) => setTokenizerName(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded text-sm mb-3"
          >
            <option value="meta-llama/Llama-3.2-3B-Instruct">meta-llama/Llama-3.2-3B-Instruct</option>
            <option value="Qwen/Qwen3-4b">Qwen/Qwen3-4b</option>
          </select>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Output Directory Path
          </label>
          <input
            type="text"
            placeholder="Output directory path"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded text-sm"
          />
        </div>
        <div className="flex flex-col gap-2">
          {console.log('Rendering datasets:', datasets.length)}
          {datasets.map((dataset, idx) => (
            <div
              key={idx}
              className={`p-3 border border-gray-300 rounded transition-colors relative ${
                loadingDatasetPath === dataset.path
                  ? 'bg-blue-100 border-blue-400 cursor-wait' 
                  : selectedDataset === dataset.path 
                    ? 'bg-blue-50 border-blue-400 cursor-pointer' 
                    : 'cursor-pointer hover:bg-gray-200'
              } ${loadingDatasetPath ? 'pointer-events-none' : ''}`}
              onClick={() => !loadingDatasetPath && loadDataset(dataset.path)}
            >
              <div className="font-bold text-sm text-gray-800">{dataset.name}</div>
              <div className="text-xs text-gray-600 my-1 truncate" title={dataset.path}>
                {dataset.relative_path}
              </div>
              <div className="flex items-center justify-between">
                <div className="text-xs text-gray-500">{dataset.size.toFixed(2)} GB</div>
                {loadingDatasetPath === dataset.path && (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="flex-1 p-4 overflow-y-auto">
        {!selectedDataset ? (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-600">
            <h2 className="text-xl font-semibold mb-2 text-gray-800">Select a dataset to explore</h2>
            <p>Choose a dataset from the sidebar to begin exploring training examples.</p>
          </div>
        ) : loadingDatasetPath ? (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-600">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
            <h2 className="text-xl font-semibold mb-2 text-gray-800">Loading Dataset...</h2>
            <p>Please wait while we load the training examples.</p>
          </div>
        ) : !selectedExample ? (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-semibold text-gray-800">Examples Overview</h2>
              <div className="text-sm text-gray-600">
                {totalExamples} total examples
              </div>
            </div>
            
            <div className="mb-4 p-3 bg-gray-50 rounded-lg border">
              <div className="flex items-center justify-between">
                <div className="flex-1 mr-3">
                  <div className="text-xs text-gray-500 mb-1">Dataset Path:</div>
                  <div className="text-sm font-mono text-gray-700 break-all">{selectedDataset}</div>
                </div>
                <button
                  onClick={() => navigator.clipboard.writeText(selectedDataset)}
                  className="px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 text-xs rounded border border-blue-300 whitespace-nowrap"
                  title="Copy path to clipboard"
                >
                  Copy
                </button>
              </div>
            </div>
            
            {examples.length > 0 && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {examples.map((example, idx) => (
                    <div
                      key={currentPage * examplesPerPage + idx}
                      className="border border-gray-300 rounded-lg p-4 cursor-pointer transition-all hover:-translate-y-1 hover:shadow-lg"
                      onClick={() => setSelectedExample(example)}
                    >
                      <div className="text-sm text-gray-600 mb-2">{example.messages.length} messages</div>
                      <div className="text-xs text-gray-500 leading-relaxed space-y-1">
                        {example.messages.filter(msg => msg.role === 'user').slice(0, 1).map((msg, idx) => (
                          <div key={idx}>
                            <span className="font-semibold text-blue-600">User:</span> {msg.content.substring(0, 80)}...
                          </div>
                        ))}
                        {example.messages.filter(msg => msg.role === 'assistant').slice(0, 1).map((msg, idx) => (
                          <div key={idx}>
                            <span className="font-semibold text-green-600">Assistant:</span> {msg.content.substring(0, 80)}...
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                
                {Math.ceil(totalExamples / examplesPerPage) > 1 && (
                  <div className="flex items-center justify-center gap-4 mt-6">
                    <button
                      onClick={() => loadDataset(selectedDataset, Math.max(0, currentPage - 1))}
                      disabled={currentPage === 0 || loadingDatasetPath}
                      className="px-4 py-2 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <span className="text-sm text-gray-600">
                      Page {currentPage + 1} of {Math.ceil(totalExamples / examplesPerPage)}
                    </span>
                    <button
                      onClick={() => loadDataset(selectedDataset, Math.min(Math.ceil(totalExamples / examplesPerPage) - 1, currentPage + 1))}
                      disabled={currentPage === Math.ceil(totalExamples / examplesPerPage) - 1 || loadingDatasetPath}
                      className="px-4 py-2 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-300">
              <div className="flex items-center gap-4">
                <button 
                  onClick={() => setSelectedExample(null)}
                  className="px-4 py-2 bg-gray-100 border border-gray-300 rounded cursor-pointer text-sm hover:bg-gray-200"
                >← Back to Overview</button>
                <h2 className="text-xl font-semibold text-gray-800">Training Example - {selectedExample.type}</h2>
              </div>
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => navigateExample('prev')}
                  className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                  title="Previous (←)"
                >←</button>
                <span className="text-sm text-gray-600">
                  {examples.findIndex(ex => ex === selectedExample) + 1} of {examples.length}
                </span>
                <button 
                  onClick={() => navigateExample('next')}
                  className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                  title="Next (→)"
                >→</button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto flex flex-col gap-4">
              {selectedExample.system_prompt && (
                <div className="p-4 rounded-lg bg-yellow-50 border border-yellow-200">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-bold text-sm text-yellow-800">System Prompt</div>
                    {selectedExample.system_prompt.length > 300 && (
                      <button
                        onClick={() => setSystemPromptExpanded(!systemPromptExpanded)}
                        className="text-xs px-2 py-1 bg-yellow-200 hover:bg-yellow-300 text-yellow-800 rounded"
                      >
                        {systemPromptExpanded ? 'Collapse' : 'Expand'}
                      </button>
                    )}
                  </div>
                  <div className="break-words overflow-wrap-anywhere leading-relaxed text-yellow-900 prose prose-sm max-w-none">
                    <ReactMarkdown>
                      {systemPromptExpanded 
                        ? selectedExample.system_prompt 
                        : selectedExample.system_prompt.length > 300 
                          ? selectedExample.system_prompt.substring(0, 300) + '...'
                          : selectedExample.system_prompt
                      }
                    </ReactMarkdown>
                  </div>
                </div>
              )}
              {selectedExample.messages.map((message, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-lg break-words ${
                    message.role === 'user' ? 'bg-blue-50 self-end max-w-[70%] ml-auto' :
                    message.role === 'assistant' ? 'bg-green-50 self-start max-w-[70%] mr-auto' :
                    'bg-orange-50 self-center max-w-[80%] mx-auto'
                  }`}
                >
                  <div className={`font-bold text-sm mb-2 capitalize ${
                    message.role === 'user' ? 'text-blue-700' :
                    message.role === 'assistant' ? 'text-green-700' :
                    'text-orange-700'
                  }`}>{message.role}</div>
                  <div className="break-words overflow-wrap-anywhere leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  </div>
                  {message.top_logprobs && (
                    <TokenVisualization message={message} />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
