import React from 'react'

interface HeaderProps {
  totalRatings: number
  isGridMode: boolean
  setIsGridMode: (value: boolean) => void
  isOpenAIEnabled: boolean
  setIsOpenAIEnabled: (value: boolean) => void
  avgSigma: number
  unresolvedItems: number
  percentFinished: number
  theoreticalMinPairs: number
}

const Header: React.FC<HeaderProps> = ({ 
  totalRatings, 
  isGridMode, 
  setIsGridMode, 
  isOpenAIEnabled, 
  setIsOpenAIEnabled,
  avgSigma,
  unresolvedItems,
  percentFinished,
  theoreticalMinPairs
}) => {
  return (
    <header className="app-header">
      <h1>Image Quality Comparison</h1>
      
      <div className="stats-container">
        <div className="stats-metrics">
          <p>Total Ratings: {totalRatings} / {theoreticalMinPairs} needed</p>
          <p title="Percentage of images with resolved ratings">Progress: {percentFinished.toFixed(1)}%</p>
          <p title="Number of images still needing more comparisons">Unresolved: {unresolvedItems}</p>
          <p title="Average uncertainty in image ratings (lower is better)">Avg Uncertainty: {avgSigma.toFixed(2)}</p>
        </div>
        
        <div className="toggle-container">
          <div className="grid-mode-toggle">
            <span className="toggle-label">Grid Mode</span>
            <label className="toggle">
              <input 
                type="checkbox" 
                checked={isGridMode}
                onChange={(e) => setIsGridMode(e.target.checked)}
              />
              <span className="slider"></span>
            </label>
          </div>
          
          <div className="openai-toggle">
            <span className="toggle-label">Auto-fetch AI Reasoning</span>
            <label className="toggle">
              <input 
                type="checkbox" 
                checked={isOpenAIEnabled}
                onChange={(e) => setIsOpenAIEnabled(e.target.checked)}
              />
              <span className="slider"></span>
            </label>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header