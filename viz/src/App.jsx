import { Routes, Route, Link, useLocation, Navigate } from 'react-router-dom'
import DatasetsPage from './pages/DatasetsPage'
import TrainingPage from './pages/TrainingPage'

function App() {
  const location = useLocation()

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ minWidth: '100vw' }}>
      {/* Navigation Header */}
      <nav className="bg-gray-800 text-white p-4 w-full flex-shrink-0">
        <div className="flex items-center justify-between w-full">
          <h1 className="text-xl font-bold">Dataset & Training Visualization</h1>
          <div className="flex gap-1">
            <Link
              to="/datasets"
              className={`px-4 py-2 rounded transition-colors ${
                location.pathname === '/datasets' 
                  ? 'bg-gray-600 text-white' 
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
              }`}
            >
              Datasets
            </Link>
            <Link
              to="/training"
              className={`px-4 py-2 rounded transition-colors ${
                location.pathname === '/training' 
                  ? 'bg-gray-600 text-white' 
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
              }`}
            >
              Training
            </Link>
          </div>
        </div>
      </nav>

      {/* Route Content */}
      <div className="flex-1 min-h-0 w-full overflow-hidden" style={{ width: '100vw' }}>
        <Routes>
          <Route path="/" element={<Navigate to="/datasets" replace />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </div>
    </div>
  )
}

export default App