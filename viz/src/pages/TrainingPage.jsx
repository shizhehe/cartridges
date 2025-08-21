import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import Plot from 'plotly.js-dist-min'

function TrainingPage() {
  // Helper functions for localStorage
  const saveToStorage = (key, value) => {
    try {
      localStorage.setItem(`training_${key}`, JSON.stringify(value))
    } catch (error) {
      console.warn('Failed to save to localStorage:', error)
    }
  }

  const loadFromStorage = (key, defaultValue) => {
    try {
      const saved = localStorage.getItem(`training_${key}`)
      return saved ? JSON.parse(saved) : defaultValue
    } catch (error) {
      console.warn('Failed to load from localStorage:', error)
      return defaultValue
    }
  }

  // W&B state - entity and project are now configured on server via env vars
  const [wandbRuns, setWandbRuns] = useState(() => loadFromStorage('wandbRuns', []))
  const [loadingWandb, setLoadingWandb] = useState(false)
  const [wandbError, setWandbError] = useState(null)
  const [wandbFilters, setWandbFilters] = useState(() => loadFromStorage('wandbFilters', {
    tag: '',
    run_id: ''
  }))
  
  // Dashboard state with persistence
  const [dashboards, setDashboards] = useState(() => loadFromStorage('dashboards', []))
  const [selectedDashboard, setSelectedDashboard] = useState(() => loadFromStorage('selectedDashboard', ''))
  const [selectedRun, setSelectedRun] = useState(() => loadFromStorage('selectedRun', null))
  const [dashboardData, setDashboardData] = useState(null)
  const [loadingDashboard, setLoadingDashboard] = useState(false)
  const [plotData, setPlotData] = useState([])
  const [loadingPlots, setLoadingPlots] = useState(false)
  const [selectedTable, setSelectedTable] = useState(null)
  const [selectedTableData, setSelectedTableData] = useState(null)
  const [loadingTableData, setLoadingTableData] = useState(false)
  const [selectedTableExample, setSelectedTableExample] = useState(null)

  // Save state to localStorage whenever it changes

  useEffect(() => {
    saveToStorage('wandbRuns', wandbRuns)
  }, [wandbRuns])

  useEffect(() => {
    saveToStorage('wandbFilters', wandbFilters)
  }, [wandbFilters])

  useEffect(() => {
    saveToStorage('dashboards', dashboards)
  }, [dashboards])

  useEffect(() => {
    saveToStorage('selectedDashboard', selectedDashboard)
  }, [selectedDashboard])

  useEffect(() => {
    saveToStorage('selectedRun', selectedRun)
  }, [selectedRun])

  // Navigate table examples (similar to dataset navigation)
  const navigateTableExample = (direction) => {
    if (!selectedTableExample || !selectedTableData || selectedTableData.length === 0) return
    
    const currentIndex = selectedTableData.findIndex(ex => ex === selectedTableExample)
    let newIndex
    
    if (direction === 'next') {
      newIndex = (currentIndex + 1) % selectedTableData.length
    } else {
      newIndex = (currentIndex - 1 + selectedTableData.length) % selectedTableData.length
    }
    
    setSelectedTableExample(selectedTableData[newIndex])
  }

  // Keyboard navigation for table examples
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!selectedTableExample) return
      
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        navigateTableExample('next')
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        navigateTableExample('prev')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedTableExample, selectedTableData])

  // W&B functionality using backend API
  const fetchWandbRuns = async () => {
    setLoadingWandb(true)
    setWandbError(null)
    
    try {
      // Get dashboard filters if a dashboard is selected
      const selectedDashboardData = dashboards.find(d => d.name === selectedDashboard)
      const dashboardFilters = selectedDashboardData ? selectedDashboardData.filters : {}
      
      console.log('Fetching W&B runs with:', { selectedDashboard, dashboardFilters, dashboards: dashboards.length })
      
      const response = await fetch('/api/wandb/runs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          filters: wandbFilters,
          dashboard_filters: dashboardFilters
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      setWandbRuns(data.runs)
    } catch (error) {
      console.error('Failed to fetch W&B runs:', error)
      setWandbError(`Failed to fetch runs: ${error.message}`)
    } finally {
      setLoadingWandb(false)
    }
  }

  // Fetch runs when dashboard selection changes OR when dashboards are first loaded
  useEffect(() => {
    if (dashboards.length > 0) {
      console.log('useEffect triggered for W&B runs fetch:', { selectedDashboard, dashboardsLength: dashboards.length })
      fetchWandbRuns()
    }
  }, [selectedDashboard, dashboards])

  // Fetch available dashboards
  const fetchDashboards = async () => {
    try {
      const response = await fetch('/api/dashboards')
      const data = await response.json()
      
      if (data.error) {
        console.error('Failed to fetch dashboards:', data.error)
        return
      }
      
      console.log('Fetched dashboards:', data.dashboards.length, 'current selectedDashboard:', selectedDashboard)
      setDashboards(data.dashboards)
      if (data.dashboards.length > 0 && !selectedDashboard) {
        console.log('Auto-selecting first dashboard:', data.dashboards[0].name)
        setSelectedDashboard(data.dashboards[0].name)
      }
    } catch (error) {
      console.error('Failed to fetch dashboards:', error)
    }
  }

  // Auto-fetch dashboards when component mounts
  useEffect(() => {
    // Only fetch dashboards if we don't have them persisted already
    if (dashboards.length === 0) {
      fetchDashboards()
    }
  }, [])

  // Debug log on mount to see restored state
  useEffect(() => {
    console.log('TrainingPage mounted with restored state:', {
      selectedDashboard,
      dashboardsCount: dashboards.length,
      runsCount: wandbRuns.length
    })
  }, [])

  // Auto-fetch runs on mount if we have all the required data persisted
  useEffect(() => {
    if (dashboards.length > 0 && wandbRuns.length === 0 && selectedDashboard) {
      console.log('Auto-fetching runs on mount with persisted data')
      fetchWandbRuns()
    }
  }, [])  // Only run on mount

  // Auto-analyze restored selected run when we have all the data
  useEffect(() => {
    if (selectedRun && selectedDashboard && dashboards.length > 0 && !dashboardData && !loadingDashboard) {
      console.log('Auto-analyzing restored selected run:', selectedRun.id)
      analyzeRun(selectedRun)
    }
  }, [selectedRun, selectedDashboard, dashboards.length, dashboardData, loadingDashboard])

  // Analyze run with selected dashboard
  const analyzeRun = async (run) => {
    if (!selectedDashboard) {
      console.error('No dashboard selected')
      return
    }

    setLoadingDashboard(true)
    setSelectedRun(run)
    setDashboardData(null)
    setPlotData([])  // Clear previous plot data
    setSelectedTable(null)
    setSelectedTableData(null)  // Clear previous table data
    setSelectedTableExample(null)
    
    try {
      const response = await fetch('/api/dashboard/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          dashboard_name: selectedDashboard
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Dashboard data loaded:', data)
      setDashboardData(data)
      
      // Start loading plots asynchronously (don't await)
      console.log('Starting async plot loading...')
      loadPlotData(run, selectedDashboard)
      
      if (data.tables.length > 0) {
        // Select the last table (highest step) by default since that's usually the final results
        const lastTable = data.tables[data.tables.length - 1]
        console.log('Loading last table by default:', lastTable)
        setSelectedTable(lastTable)
        loadTableData(lastTable, run)  // Load the last table's data, passing run directly
      } else {
        console.log('No tables found in dashboard data')
      }
    } catch (error) {
      console.error('Failed to analyze run:', error)
      setWandbError(`Failed to analyze run: ${error.message}`)
    } finally {
      setLoadingDashboard(false)
    }
  }

  // Load plot data asynchronously
  const loadPlotData = async (run = selectedRun, dashboard = selectedDashboard) => {
    console.log('loadPlotData called', { run, dashboard })
    if (!run || !dashboard) {
      console.log('Missing run or dashboard, skipping plot loading')
      return
    }

    setLoadingPlots(true)
    console.log('Loading plots for run:', run.id, 'dashboard:', dashboard)
    
    try {
      const response = await fetch('/api/dashboard/plots', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          dashboard_name: dashboard
        })
      })
      
      const data = await response.json()
      console.log('Plot data response:', data)
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Setting plot data:', data.plots)
      setPlotData(data.plots)
    } catch (error) {
      console.error('Failed to load plot data:', error)
      setWandbError(`Failed to load plot data: ${error.message}`)
    } finally {
      setLoadingPlots(false)
    }
  }

  // Load table data on demand
  const loadTableData = async (table, run = selectedRun) => {
    console.log('loadTableData called with:', { run: run?.id, table })
    if (!run || !table) {
      console.log('Missing run or table, skipping table data loading')
      return
    }

    setLoadingTableData(true)
    console.log('Loading table data for step:', table.step)
    
    try {
      const response = await fetch('/api/dashboard/table', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          run_id: run.id,
          table_path: table.path,
          table_step: table.step
        })
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      console.log('Table data loaded successfully:', data.data.length, 'rows')
      setSelectedTableData(data.data)
    } catch (error) {
      console.error('Failed to load table data:', error)
      setWandbError(`Failed to load table data: ${error.message}`)
    } finally {
      setLoadingTableData(false)
    }
  }

  // Plot visualization component
  const PlotVisualization = ({ plot }) => {
    const plotRef = useRef(null)
    
    useEffect(() => {
      if (!plotRef.current || !plot) return
      
      const trace = {
        x: plot.data.map(d => d[plot.x_col]),
        y: plot.data.map(d => d[plot.y_col]),
        type: 'scatter',
        mode: 'lines+markers',
        name: plot.plot_name,
        line: { color: '#3b82f6' },
        marker: { color: '#3b82f6', size: 6 }
      }
      
      const layout = {
        title: {
          text: plot.plot_name,
          font: { size: 14 }
        },
        xaxis: { 
          title: plot.x_col,
          gridcolor: '#f3f4f6'
        },
        yaxis: { 
          title: plot.y_col,
          gridcolor: '#f3f4f6'
        },
        margin: { l: 50, r: 30, t: 40, b: 40 },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
      }
      
      const config = {
        displayModeBar: false,
        responsive: true
      }
      
      Plot.newPlot(plotRef.current, [trace], layout, config)
    }, [plot])
    
    if (!plot) return null
    
    return <div ref={plotRef} className="w-full h-80" />
  }

  return (
    <div className="flex h-full font-sans overflow-hidden" style={{ width: '100vw', minWidth: '100vw' }}>
      <div className="w-96 bg-gray-100 border-r border-gray-300 flex flex-col flex-shrink-0 overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-gray-300">
          <h1 className="text-xl font-bold text-gray-800">Training Dashboard</h1>
        </div>

        {/* Dashboard Selection */}
        <div className="p-4 border-b border-gray-300">
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dashboard
              </label>
              <select
                value={selectedDashboard}
                onChange={(e) => setSelectedDashboard(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded text-sm"
              >
                <option value="">Select a dashboard</option>
                {dashboards.map((dashboard) => (
                  <option key={dashboard.name} value={dashboard.name}>
                    {dashboard.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* W&B Configuration */}
        <div className="p-4">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Filter by Tag
              </label>
              <input
                type="text"
                value={wandbFilters.tag}
                onChange={(e) => setWandbFilters({...wandbFilters, tag: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    fetchWandbRuns()
                  }
                }}
                placeholder="e.g., generate, paper"
                className="w-full p-2 border border-gray-300 rounded text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Filter by Run ID/Name
              </label>
              <input
                type="text"
                value={wandbFilters.run_id}
                onChange={(e) => setWandbFilters({...wandbFilters, run_id: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    fetchWandbRuns()
                  }
                }}
                placeholder="e.g., longhealth, p10"
                className="w-full p-2 border border-gray-300 rounded text-sm"
              />
            </div>
          </div>
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
        {(wandbRuns.length > 0 || (!loadingWandb && dashboards.length > 0)) && (
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-3 px-4">
              <h3 className="text-sm font-medium text-gray-700">
                Recent Runs ({wandbRuns.length})
              </h3>
              <button
                onClick={fetchWandbRuns}
                disabled={loadingWandb}
                className={`px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400 text-gray-700 rounded transition-colors ${
                  loadingWandb ? 'animate-spin' : ''
                }`}
                title="Refresh runs"
              >
                ↻
              </button>
            </div>
            <div className="space-y-2 flex-1 overflow-y-auto px-4">
              {wandbRuns.length > 0 ? wandbRuns.map((run) => (
                <div
                  key={run.id}
                  className={`p-3 rounded-lg cursor-pointer transition-all border ${
                    selectedRun?.id === run.id ? 'bg-blue-50 border-blue-300' : ''
                  }`}
                  onClick={() => analyzeRun(run)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-medium text-sm text-gray-800 truncate">
                      {run.name}
                    </div>
                    <div className={`px-2 py-1 text-xs rounded-full ${
                      run.state === 'running' ? 'bg-blue-100 text-blue-800' :
                      run.state === 'finished' ? 'bg-green-100 text-green-800' :
                      run.state === 'failed' ? 'bg-red-100 text-red-800' :
                      run.state === 'crashed' ? 'bg-red-100 text-red-800 bg-red-200' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {run.state}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 mb-1">
                    ID: {run.id}
                  </div>
                  {run.tags && run.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {run.tags.slice(0, 3).map((tag, idx) => (
                        <span key={idx} className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded">
                          {tag}
                        </span>
                      ))}
                      {run.tags.length > 3 && (
                        <span className="text-xs text-gray-500">+{run.tags.length - 3} more</span>
                      )}
                    </div>
                  )}
                  <div className="text-xs text-gray-400">
                    Created: {new Date(run.createdAt).toLocaleDateString()}
                  </div>
                  <a 
                    href={run.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-xs text-blue-600 hover:text-blue-800 mt-1 inline-block"
                    onClick={(e) => e.stopPropagation()}
                  >
                    View in W&B →
                  </a>
                </div>
              )) : (
                <div className="text-center text-gray-500 py-8">
                  <p className="text-sm">No runs found.</p>
                  <p className="text-xs mt-1">Use filters above and click refresh to fetch runs.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* W&B API Configuration Note */}
        <div className="p-4 mt-auto">
          <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded">
            W&B API Key, Entity, and Project are configured on the server via environment variables.
          </div>
        </div>
      </div>
      
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden" style={{ width: 'calc(100vw - 24rem)' }}>
        {selectedRun && dashboardData ? (
          <div className="flex flex-col h-full w-full overflow-hidden">
            {/* Dashboard Header */}
            <div className="bg-white p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xl font-semibold text-gray-800">
                  {selectedRun.name} - {selectedDashboard}
                </h2>
                <span className="text-sm text-gray-500">
                  Run ID: {selectedRun.id}
                </span>
              </div>
            </div>

            {/* Plots Section */}
            {(dashboardData?.plots?.length > 0 || plotData.length > 0 || loadingPlots) && (
              <div className="bg-white p-4 border-b border-gray-200 w-full flex-shrink-0">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Metrics</h3>
                {loadingPlots && plotData.length === 0 ? (
                  <div className="flex items-center justify-center p-8 w-full">
                    <div className="text-gray-500">Loading plots...</div>
                  </div>
                ) : (
                  <div className="w-full">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full">
                      {plotData.map((plot, idx) => (
                        <div key={idx} className="border border-gray-200 rounded-lg p-4 w-full min-w-0">
                          <PlotVisualization plot={plot} />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Tables Section */}
            {dashboardData.tables && dashboardData.tables.length > 0 && (
              <div className="flex-1 flex flex-col min-h-0 w-full overflow-hidden">
                <div className="bg-white p-4 border-b border-gray-200 w-full flex-shrink-0">
                  <div className="flex items-center justify-between w-full">
                    <h3 className="text-lg font-semibold text-gray-800">Tables</h3>
                    <select
                      value={selectedTable ? dashboardData.tables.indexOf(selectedTable) : 0}
                      onChange={(e) => {
                        const newTable = dashboardData.tables[parseInt(e.target.value)]
                        setSelectedTable(newTable)
                        setSelectedTableData(null)  // Clear previous data
                        setSelectedTableExample(null)  // Clear selected example
                        loadTableData(newTable)  // Load new table data
                      }}
                      className="p-2 border border-gray-300 rounded text-sm flex-shrink-0"
                    >
                      {dashboardData.tables.map((table, idx) => (
                        <option key={idx} value={idx}>
                          Step {table.step}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {!selectedTableExample ? (
                  /* Table Gallery View */
                  <div className="flex-1 overflow-y-auto p-4 w-full">
                    {loadingTableData ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center text-gray-600 w-full">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                        <p>Loading table data...</p>
                      </div>
                    ) : selectedTableData && selectedTableData.length > 0 ? (
                      <div className="w-full">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full">
                          {selectedTableData.map((example, idx) => (
                            <div
                              key={idx}
                              className="border border-gray-300 rounded-lg p-4 cursor-pointer transition-all hover:-translate-y-1 hover:shadow-lg w-full min-h-[200px]"
                              onClick={() => setSelectedTableExample(example)}
                            >
                              <div className="text-sm text-gray-600 mb-2">
                                Score: {example[selectedTable.score_col] || 'N/A'}
                              </div>
                              <div className="text-xs text-gray-500 leading-relaxed space-y-1">
                                <div className="truncate">
                                  <span className="font-semibold text-blue-600">Prompt:</span>{' '}
                                  {(example[selectedTable.prompt_col] || '').substring(0, 80)}...
                                </div>
                                <div className="truncate">
                                  <span className="font-semibold text-green-600">Answer:</span>{' '}
                                  {(example[selectedTable.answer_col] || '').substring(0, 80)}...
                                </div>
                                <div className="truncate">
                                  <span className="font-semibold text-purple-600">Prediction:</span>{' '}
                                  {(example[selectedTable.pred_col] || '').substring(0, 80)}...
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : selectedTable ? (
                      <div className="text-center text-gray-500 py-12 w-full">
                        No table data available for this step.
                      </div>
                    ) : (
                      <div className="text-center text-gray-500 py-12 w-full">
                        Select a table step to view data.
                      </div>
                    )}
                  </div>
                ) : (
                  /* Table Focus Mode */
                  <div className="flex-1 flex flex-col w-full overflow-hidden">
                    <div className="flex items-center justify-between mb-4 pb-4 pt-6 border-b border-gray-300 px-6 w-full flex-shrink-0">
                      <button 
                        onClick={() => setSelectedTableExample(null)}
                        className="px-4 py-2 bg-gray-100 border border-gray-300 rounded cursor-pointer text-sm hover:bg-gray-200"
                      >← Back to Table</button>
                      <h2 className="text-xl font-semibold text-gray-800">
                        Table Example - Step {selectedTable.step}
                      </h2>
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => navigateTableExample('prev')}
                          className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                          title="Previous (←)"
                        >←</button>
                        <span className="text-sm text-gray-600">
                          {selectedTableData.findIndex(ex => ex === selectedTableExample) + 1} of {selectedTableData.length}
                        </span>
                        <button 
                          onClick={() => navigateTableExample('next')}
                          className="px-3 py-1 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200"
                          title="Next (→)"
                        >→</button>
                      </div>
                    </div>
                    <div className="flex-1 overflow-y-auto px-6 w-full">
                      <div className="space-y-6">
                        <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
                          <div className="font-bold text-sm text-blue-800 mb-2">Prompt</div>
                          <div className="break-words leading-relaxed text-blue-900 prose prose-sm max-w-none">
                            <ReactMarkdown>{selectedTableExample[selectedTable.prompt_col] || 'No prompt'}</ReactMarkdown>
                          </div>
                        </div>
                        <div className="p-4 rounded-lg bg-green-50 border border-green-200">
                          <div className="font-bold text-sm text-green-800 mb-2">Expected Answer</div>
                          <div className="break-words leading-relaxed text-green-900 prose prose-sm max-w-none">
                            <ReactMarkdown>{selectedTableExample[selectedTable.answer_col] || 'No answer'}</ReactMarkdown>
                          </div>
                        </div>
                        <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
                          <div className="font-bold text-sm text-purple-800 mb-2">Prediction</div>
                          <div className="break-words leading-relaxed text-purple-900 prose prose-sm max-w-none">
                            <ReactMarkdown>{selectedTableExample[selectedTable.pred_col] || 'No prediction'}</ReactMarkdown>
                          </div>
                        </div>
                        <div className="p-4 rounded-lg bg-gray-50 border border-gray-200">
                          <div className="font-bold text-sm text-gray-800 mb-2">Score</div>
                          <div className="text-gray-900">
                            {selectedTableExample[selectedTable.score_col] || 'No score'}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Loading State */}
            {loadingDashboard && (
              <div className="flex flex-col items-center justify-center py-12 text-center text-gray-600">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                <p>Loading run from W&B...</p>
              </div>
            )}
          </div>
        ) : (
          loadingDashboard ? (
            <div className="flex flex-col items-center justify-center text-center text-gray-600 flex-1 p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-4"></div>
              <h2 className="text-xl font-semibold mb-2 text-gray-800">Loading run from W&B...</h2>
              <p>Loading dashboard data and metrics.</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-center text-gray-600 flex-1 p-4">
              <h2 className="text-xl font-semibold mb-2 text-gray-800">Select a run to analyze</h2>
              <p>Choose a dashboard and click on a W&B run to begin analysis.</p>
            </div>
          )
        )}
      </div>
    </div>
  )
}

export default TrainingPage