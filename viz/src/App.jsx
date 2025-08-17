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
  const [datasetError, setDatasetError] = useState(null)
  const [configData, setConfigData] = useState(null)
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [activeTab, setActiveTab] = useState('datasets')
  
  // W&B state
  const wandbApiKey = import.meta.env.VITE_WANDB_API_KEY || ''
  const [wandbEntity, setWandbEntity] = useState('hazy-research')
  const [wandbProject, setWandbProject] = useState('cartridges')
  const [wandbRuns, setWandbRuns] = useState([])
  const [loadingWandb, setLoadingWandb] = useState(false)
  const [wandbError, setWandbError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchFields, setSearchFields] = useState({
    messages: true,
    system_prompt: false,
    metadata: false
  })
  const [isScrolled, setIsScrolled] = useState(false)

  // Dataset discovery
  useEffect(() => {
    const discoverDatasets = async () => {
      try {
        setLoadingDatasets(true)
        console.log('Fetching datasets...')
        const response = await fetch('/api/datasets')
        const data = await response.json()
        console.log('Datasets received:', data)
        // Ensure datasets are sorted by relative path (reverse alphabetical)
        const sortedData = data.sort((a, b) => b.relative_path.localeCompare(a.relative_path))
        setDatasets(sortedData)
      } catch (error) {
        console.error('Failed to discover datasets:', error)
      } finally {
        setLoadingDatasets(false)
      }
    }
    discoverDatasets()
  }, [outputDir])

  const selectDataset = async (datasetPath) => {
    // Immediately set the selected dataset and show the path
    setSelectedDataset(datasetPath)
    setExamples([])
    setSelectedExample(null)
    setCurrentPage(0)
    setLoadingDatasetPath(datasetPath)
    setDatasetError(null)
    setConfigData(null)
    setSearchQuery('') // Reset search when switching datasets
    
    try {
      // First, load dataset metadata quickly
      const infoResponse = await fetch(`/api/dataset/${encodeURIComponent(datasetPath)}/info`)
      const info = await infoResponse.json()
      setTotalExamples(info.total_count)
      
      // Then load the first page of examples (will be updated by useEffect)
      await loadDatasetWithSearch(0)
      
      // Also automatically load the config
      try {
        const configResponse = await fetch('/api/dataset/config', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            dataset_path: datasetPath
          })
        })
        const configData = await configResponse.json()
        setConfigData(configData)
      } catch (configError) {
        console.error('Failed to load config:', configError)
        setConfigData({ error: 'Failed to load config', exists: false })
      }
    } catch (error) {
      console.error('Failed to load dataset:', error)
      // Keep dataset selected but show error state
      setDatasetError(error.message || 'Failed to load dataset')
      setTotalExamples(0)
      setExamples([])
    } finally {
      setLoadingDatasetPath(null)
    }
  }

  // W&B functionality using REST API
  const fetchWandbRuns = async () => {
    if (!wandbApiKey || !wandbEntity || !wandbProject) {
      setWandbError('Please configure W&B API key, entity, and project')
      return
    }

    setLoadingWandb(true)
    setWandbError(null)
    
    try {
      console.log('API Key length:', wandbApiKey.length)
      console.log('Making request to:', `https://api.wandb.ai/api/v1/runs/${wandbEntity}/${wandbProject}`)
      // Use W&B REST API directly
      const response = await fetch(`https://api.wandb.ai/api/v1/runs/${wandbEntity}/${wandbProject}`, {
        headers: {
          'Authorization': `Bearer ${wandbApiKey}`,
          'Content-Type': 'application/json'
        }
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      const runData = data.runs.map(run => ({
        id: run.id,
        name: run.name || run.id,
        state: run.state,
        createdAt: run.createdAt,
        config: run.config,
        summary: run.summary,
        tags: run.tags || [],
        url: `https://wandb.ai/${wandbEntity}/${wandbProject}/runs/${run.id}`
      }))
      
      setWandbRuns(runData)
    } catch (error) {
      console.error('Failed to fetch W&B runs:', error)
      setWandbError(`Failed to fetch runs: ${error.message}`)
    } finally {
      setLoadingWandb(false)
    }
  }

  // Auto-fetch runs when tab becomes active and credentials are set
  useEffect(() => {
    if (activeTab === 'training' && wandbApiKey && wandbEntity && wandbProject) {
      fetchWandbRuns()
    }
  }, [activeTab, wandbEntity, wandbProject])

  // Load dataset with search
  const loadDatasetWithSearch = async (page = 0) => {
    if (!selectedDataset) return
    
    setLoadingDatasetPath(selectedDataset)
    setDatasetError(null)
    
    try {
      const searchParams = new URLSearchParams({
        page: page.toString(),
        page_size: examplesPerPage.toString()
      })
      
      if (searchQuery.trim()) {
        searchParams.set('search', searchQuery)
        searchParams.set('search_messages', searchFields.messages ? 'true' : 'false')
        searchParams.set('search_system_prompt', searchFields.system_prompt ? 'true' : 'false')
        searchParams.set('search_metadata', searchFields.metadata ? 'true' : 'false')
        console.log('Search params:', { search: searchQuery, fields: searchFields })
      }
      
      const response = await fetch(`/api/dataset/${encodeURIComponent(selectedDataset)}?${searchParams}`)
      const data = await response.json()
      setExamples(data.examples)
      setTotalExamples(data.total_count)
      setCurrentPage(page)
    } catch (error) {
      console.error('Failed to load dataset:', error)
      setDatasetError(error.message || 'Failed to load dataset')
      setTotalExamples(0)
      setExamples([])
    } finally {
      setLoadingDatasetPath(null)
    }
  }

  // Reload data when search query or search fields change
  useEffect(() => {
    if (selectedDataset) {
      loadDatasetWithSearch(0) // Reset to first page when search changes
    }
  }, [searchQuery, searchFields])

  // No need for client-side filtering - examples are already filtered by backend
  const filteredExamples = examples

  const loadDatasetPage = async (page) => {
    loadDatasetWithSearch(page)
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

  // Collapsible JSON Config Component
  const ConfigViewer = () => {
    if (!configData) return null

    const [codeExpanded, setCodeExpanded] = useState(false)

    return (
      <div className="mb-6 bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="bg-gradient-to-r from-purple-500 to-purple-600 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-white rounded-full opacity-80"></div>
              <div className="font-semibold text-white text-sm font-mono">SynthesizeConfig</div>
            </div>
            <button
              onClick={() => setCodeExpanded(!codeExpanded)}
              className="text-xs px-3 py-1.5 bg-white/20 hover:bg-white/30 text-white rounded-md transition-colors flex items-center gap-1"
            >
              <svg 
                className={`w-3 h-3 transition-transform ${codeExpanded ? 'rotate-180' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
              {codeExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
        </div>
        
        {codeExpanded && (
          <div className="p-0">
            {configData.exists && configData.config ? (
              <div className="bg-gray-950 text-gray-100">
                {/* Simple header with copy button */}
                <div className="flex items-center justify-between px-4 py-3 bg-gray-900 border-b border-gray-700">
                  <div className="text-xs text-gray-400 font-mono break-all">
                    {configData.path || 'config.yaml'}
                  </div>
                  <button
                    onClick={() => navigator.clipboard.writeText(JSON.stringify(configData.config, null, 2))}
                    className="text-xs px-3 py-1.5 bg-purple-600 hover:bg-purple-700 text-white rounded-md transition-colors"
                    title="Copy JSON to clipboard"
                  >
                    Copy JSON
                  </button>
                </div>
                
                {/* JSON content */}
                <div className="p-4 overflow-auto max-h-96 max-w-full">
                  <div className="overflow-x-auto">
                    <pre className="text-sm leading-relaxed whitespace-pre-wrap min-w-0">
                      <code className="language-json">
                        {JSON.stringify(configData.config, null, 2)
                          .split('\n')
                          .map((line, index) => (
                            <div key={index} className="flex min-w-0">
                              <span className="flex-shrink-0 text-right pr-4 text-gray-500 select-none text-xs leading-relaxed w-12">
                                {index + 1}
                              </span>
                              <span className="text-green-300 leading-relaxed break-all min-w-0 flex-1">
                                {line}
                              </span>
                            </div>
                          ))
                        }
                      </code>
                    </pre>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center">
                <div className="text-gray-400 text-4xl mb-3">📄</div>
                <div className="text-gray-600 font-medium mb-1">No Configuration Found</div>
                <div className="text-sm text-gray-500">
                  {configData.error || 'No config.yaml file found for this dataset'}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-screen font-sans">
      <div className="w-80 bg-gray-100 border-r border-gray-300 flex flex-col">
        {/* Tab Navigation */}
        <div className="flex gap-1 bg-gray-100 p-2">
          <button
            onClick={() => setActiveTab('datasets')}
            className={`flex-1 px-4 py-2.5 text-sm font-medium rounded-t-lg border-t border-l border-r transition-all duration-200 ${
              activeTab === 'datasets'
                ? 'bg-white text-gray-800 border-gray-300 shadow-sm'
                : 'bg-gray-100 text-gray-600 border-transparent hover:bg-gray-200'
            }`}
          >
            Datasets
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`flex-1 px-4 py-2.5 text-sm font-medium rounded-t-lg border-t border-l border-r transition-all duration-200 ${
              activeTab === 'training'
                ? 'bg-white text-gray-800 border-gray-300 shadow-sm'
                : 'bg-gray-100 text-gray-600 border-transparent hover:bg-gray-200'
            }`}
          >
            Training
          </button>
        </div>
        
        {/* Tab Content */}
        <div className="flex-1 p-4 overflow-y-auto">
          {activeTab === 'datasets' && (
            <>
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
                {loadingDatasets ? (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mb-3"></div>
                    <div className="text-sm text-gray-600">Loading datasets...</div>
                  </div>
                ) : (
                  <>
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
                        onClick={() => !loadingDatasetPath && selectDataset(dataset.path)}
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
                  </>
                )}
              </div>
            </>
          )}
          
          {activeTab === 'training' && (
            <div className="space-y-4">
              {/* W&B Configuration */}
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    W&B API Key
                  </label>
                  <input
                    type="password"
                    placeholder="Using environment variable..."
                    value={wandbApiKey ? '••••••••••••••••' : ''}
                    disabled={true}
                    className="w-full p-2 border border-gray-300 rounded text-sm bg-gray-100"
                  />
                  {!wandbApiKey && (
                    <div className="text-xs text-red-600 mt-1">
                      WANDB_API_KEY environment variable not set
                    </div>
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Entity
                    </label>
                    <input
                      type="text"
                      placeholder="hazy-research"
                      value={wandbEntity}
                      onChange={(e) => setWandbEntity(e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Project
                    </label>
                    <input
                      type="text"
                      placeholder="cartridges"
                      value={wandbProject}
                      onChange={(e) => setWandbProject(e.target.value)}
                      className="w-full p-2 border border-gray-300 rounded text-sm"
                    />
                  </div>
                </div>
                
                <button
                  onClick={fetchWandbRuns}
                  disabled={loadingWandb || !wandbApiKey}
                  className="w-full p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white text-sm rounded transition-colors"
                >
                  {loadingWandb ? 'Loading Runs...' : 'Fetch W&B Runs'}
                </button>
              </div>

              {/* Error Display */}
              {wandbError && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="text-sm text-red-800">{wandbError}</div>
                </div>
              )}

              {/* Loading State */}
              {loadingWandb && (
                <div className="flex flex-col items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-3"></div>
                  <div className="text-sm text-gray-600">Fetching runs from W&B...</div>
                </div>
              )}

              {/* W&B Runs */}
              {wandbRuns.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">
                    Recent Runs ({wandbRuns.length})
                  </h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {wandbRuns.map((run) => (
                      <div key={run.id} className="p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                        <div className="flex items-center justify-between mb-2">
                          <div className="font-medium text-sm text-gray-800 truncate">
                            {run.name}
                          </div>
                          <div className={`px-2 py-1 text-xs rounded-full ${
                            run.state === 'running' ? 'bg-blue-100 text-blue-800' :
                            run.state === 'finished' ? 'bg-green-100 text-green-800' :
                            run.state === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {run.state}
                          </div>
                        </div>
                        
                        <div className="text-xs text-gray-500 space-y-1">
                          <div>ID: {run.id}</div>
                          {run.summary && Object.keys(run.summary).length > 0 && (
                            <div>
                              {Object.entries(run.summary).slice(0, 2).map(([key, value]) => (
                                <span key={key} className="mr-3">
                                  {key}: {typeof value === 'number' ? value.toFixed(3) : value}
                                </span>
                              ))}
                            </div>
                          )}
                          {run.tags && run.tags.length > 0 && (
                            <div className="flex gap-1 flex-wrap">
                              {run.tags.slice(0, 3).map((tag, idx) => (
                                <span key={idx} className="px-1 py-0.5 bg-gray-100 rounded text-xs">
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                          <div>Created: {new Date(run.createdAt).toLocaleString()}</div>
                        </div>
                        
                        {run.url && (
                          <a
                            href={run.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-blue-600 hover:text-blue-800 mt-2 inline-block"
                            onClick={(e) => e.stopPropagation()}
                          >
                            View in W&B →
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Empty State */}
              {!loadingWandb && wandbRuns.length === 0 && wandbApiKey && (
                <div className="text-center py-8 text-gray-500">
                  <div className="text-2xl mb-2">📊</div>
                  <div className="text-sm">No runs found in this project</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      <div className="flex-1 flex flex-col min-w-0">
        {!selectedDataset ? (
          <div className="flex flex-col items-center justify-center text-center text-gray-600 flex-1 p-4">
            <h2 className="text-xl font-semibold mb-2 text-gray-800">Select a dataset to explore</h2>
            <p>Choose a dataset from the sidebar to begin exploring training examples.</p>
          </div>
        ) : !selectedExample ? (
          <>
            <div className={`bg-white p-4 transition-all duration-200 ${isScrolled ? 'border-b border-gray-200 shadow-sm' : ''}`}>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-semibold text-gray-800">Examples Overview</h2>
                <div className="text-sm text-gray-600">
                  {totalExamples > 0 ? `${totalExamples} total examples` : 'Loading count...'}
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
              
              <ConfigViewer />
              
              {/* Search Input */}
              <div className="mb-4">
                <div className="flex items-center gap-4">
                  <div className="relative flex-1">
                    <input
                      type="text"
                      placeholder="Search examples by content, system prompt, or metadata..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full p-3 pl-10 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                    </div>
                    {searchQuery && (
                      <button
                        onClick={() => setSearchQuery('')}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center bg-transparent border-none"
                        title="Clear search"
                      >
                        <svg className="h-4 w-4 text-gray-500 hover:text-gray-700 transition-colors" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </button>
                    )}
                  </div>
                  <div className="flex gap-1.5 items-center">
                    {[
                      { key: 'messages', label: 'Messages' },
                      { key: 'system_prompt', label: 'System Prompt' },
                      { key: 'metadata', label: 'Metadata' }
                    ].map(({ key, label }) => (
                      <button
                        key={key}
                        onClick={() => setSearchFields(prev => ({ ...prev, [key]: !prev[key] }))}
                        className={`px-2 py-1 text-xs rounded border transition-all ${
                          searchFields[key]
                            ? 'bg-blue-100 border-blue-300 text-blue-800 shadow-sm'
                            : 'bg-gray-100 border-gray-300 text-gray-600 hover:bg-gray-200'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  {searchQuery && (
                    <div className="text-xs text-gray-600">
                      {totalExamples} results
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div 
              className="flex-1 overflow-y-auto p-4"
              onScroll={(e) => {
                const scrolled = e.target.scrollTop > 0
                if (scrolled !== isScrolled) {
                  setIsScrolled(scrolled)
                }
              }}
            >
              {datasetError ? (
                <div className="flex flex-col items-center justify-center py-12 text-center text-red-600">
                  <div className="text-red-500 mb-4">⚠️</div>
                  <h3 className="font-semibold mb-2">Error loading dataset</h3>
                  <p className="text-sm text-gray-600">{datasetError}</p>
                  <button 
                    onClick={() => selectDataset(selectedDataset)}
                    className="mt-4 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 text-sm rounded border border-blue-300"
                  >
                    Retry
                  </button>
                </div>
              ) : loadingDatasetPath && examples.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center text-gray-600">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                  <p>Loading examples...</p>
                </div>
              ) : filteredExamples.length > 0 ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filteredExamples.map((example, idx) => (
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
                        onClick={() => loadDatasetPage(Math.max(0, currentPage - 1))}
                        disabled={currentPage === 0 || loadingDatasetPath}
                        className="px-4 py-2 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Previous
                      </button>
                      <span className="text-sm text-gray-600">
                        Page {currentPage + 1} of {Math.ceil(totalExamples / examplesPerPage)}
                      </span>
                      <button
                        onClick={() => loadDatasetPage(Math.min(Math.ceil(totalExamples / examplesPerPage) - 1, currentPage + 1))}
                        disabled={currentPage === Math.ceil(totalExamples / examplesPerPage) - 1 || loadingDatasetPath}
                        className="px-4 py-2 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Next
                      </button>
                    </div>
                  )}
                </>
              ) : examples.length > 0 ? (
                <div className="text-center text-gray-500 py-12">
                  <div className="text-4xl mb-3">🔍</div>
                  <div className="font-medium mb-1">No examples match your search</div>
                  <div className="text-sm">Try a different search term or clear your search</div>
                  {searchQuery && (
                    <button 
                      onClick={() => setSearchQuery('')}
                      className="mt-3 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 text-sm rounded border border-blue-300"
                    >
                      Clear Search
                    </button>
                  )}
                </div>
              ) : (
                <div className="text-center text-gray-500 py-12">
                  No examples found in this dataset.
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-4 pb-4 pt-6 border-b border-gray-300 px-6">
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
            <div className="flex-1 overflow-y-auto px-6">
              <div className="flex flex-col gap-4">
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
          </div>
        )}
      </div>
    </div>
  )
}

export default App
