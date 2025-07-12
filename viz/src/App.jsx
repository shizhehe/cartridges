import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

function App() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [examples, setExamples] = useState([])
  const [selectedExample, setSelectedExample] = useState(null)
  const [outputDir, setOutputDir] = useState('')
  const [tokenizerName, setTokenizerName] = useState('meta-llama/Llama-3.2-3B-Instruct')

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
      if (prob < 0.1) return 'bg-red-200 hover:bg-red-300'
      if (prob < 0.5) return 'bg-yellow-200 hover:bg-yellow-300'
      return 'bg-green-200 hover:bg-green-300'
    }
    
    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold mb-2 text-gray-800">Token Probabilities (hover for details)</h4>
        <div className="relative">
          <div className="font-mono text-sm leading-relaxed">
            {tokens.map((token, idx) => (
              <span
                key={idx}
                className={`inline-block px-1 py-0.5 m-0.5 rounded cursor-pointer transition-all ${getColorForProbability(token.prob)}`}
                onMouseEnter={() => setHoveredToken(token)}
                onMouseLeave={() => setHoveredToken(null)}
                title={`${token.text} (${(token.prob * 100).toFixed(1)}%)`}
              >
                {token.text}
              </span>
            ))}
          </div>
          
          {hoveredToken && (
            <div className="absolute z-10 bg-gray-800 text-white p-3 rounded-lg shadow-lg max-w-sm" 
                 style={{
                   left: '50%',
                   top: '100%',
                   transform: 'translateX(-50%)',
                   marginTop: '8px'
                 }}>
              <div className="font-semibold mb-2">
                Token {hoveredToken.position + 1}: "{hoveredToken.text}"
              </div>
              <div className="text-sm space-y-1">
                <div>Probability: {(hoveredToken.prob * 100).toFixed(2)}%</div>
                <div>Log Probability: {hoveredToken.logprob.toFixed(4)}</div>
                <div>Entropy: {hoveredToken.entropy.toFixed(3)}</div>
              </div>
              <div className="mt-2 pt-2 border-t border-gray-600">
                <div className="text-xs font-semibold mb-1">Top Alternatives:</div>
                {hoveredToken.alternatives.map((alt, idx) => (
                  <div key={idx} className={`text-xs ${alt.chosen ? 'text-green-300 font-semibold' : 'text-gray-300'}`}>
                    {alt.rank}. "{alt.text}" ({(alt.prob * 100).toFixed(1)}%)
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen font-sans">
      <div className="w-80 bg-gray-100 border-r border-gray-300 p-4 overflow-y-auto">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">Datasets</h2>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Tokenizer Name
          </label>
          <input
            type="text"
            placeholder="e.g., meta-llama/Llama-3.2-3B-Instruct"
            value={tokenizerName}
            onChange={(e) => setTokenizerName(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded text-sm mb-3"
          />
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
          {datasets.map((dataset, idx) => (
            <div
              key={idx}
              className={`p-3 border border-gray-300 rounded cursor-pointer transition-colors hover:bg-gray-200 ${
                selectedDataset === dataset.path ? 'bg-blue-50 border-blue-400' : ''
              }`}
              onClick={() => loadDataset(dataset.path)}
            >
              <div className="font-bold text-sm text-gray-800">{dataset.name}</div>
              <div className="text-xs text-gray-600 my-1 truncate" title={dataset.path}>
                {dataset.relative_path}
              </div>
              <div className="text-xs text-gray-500">{dataset.size} examples</div>
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
        ) : !selectedExample ? (
          <div>
            <h2 className="text-2xl font-semibold mb-4 text-gray-800">Examples Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {examples.map((example, idx) => (
                <div
                  key={idx}
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
                  <div className="font-bold text-sm mb-2 text-yellow-800">System Prompt</div>
                  <div className="break-words overflow-wrap-anywhere leading-relaxed text-yellow-900 prose prose-sm max-w-none">
                    <ReactMarkdown>{selectedExample.system_prompt}</ReactMarkdown>
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
