import { useState, useEffect } from 'react'
import './App.css'
import ImageComparison from './components/ImageComparison'
import AIScorePanel from './components/AIScorePanel'
import { ImagePair } from './types'
import Header from './components/Header'
import KeyboardShortcutsGuide from './components/KeyboardShortcutsGuide'
import { fetchRandomPair, submitRating, fetchOpenAIScore } from './utils/api'

function App() {
  const [imagePair, setImagePair] = useState<ImagePair | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [totalRatings, setTotalRatings] = useState<number>(0)
  const [aiScore, setAIScore] = useState<number>(0)
  const [isGridMode, setIsGridMode] = useState<boolean>(false)
  const [openAIScore, setOpenAIScore] = useState<{ score: number, reasoning: string } | null>(null)
  const [openAILoading, setOpenAILoading] = useState<boolean>(false)
  const [isOpenAIEnabled, setIsOpenAIEnabled] = useState<boolean>(() => {
    // Load from localStorage with default of true
    const saved = localStorage.getItem('isOpenAIEnabled')
    return saved !== null ? JSON.parse(saved) : true
  })
  
  // Add state for the new metrics
  const [avgSigma, setAvgSigma] = useState<number>(0)
  const [unresolvedItems, setUnresolvedItems] = useState<number>(0)
  const [percentFinished, setPercentFinished] = useState<number>(0)
  const [theoreticalMinPairs, setTheoreticalMinPairs] = useState<number>(0)

  // Save preference to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('isOpenAIEnabled', JSON.stringify(isOpenAIEnabled))
  }, [isOpenAIEnabled])

  const getRandomPair = async () => {
    setIsLoading(true)
    setOpenAIScore(null)
    try {
      const data = await fetchRandomPair(isGridMode)
      setImagePair({
        image1: data.image1,
        image2: data.image2
      })
      setTotalRatings(data.total_ratings)
      setAIScore(data.score)
      
      // Store the new metrics
      setAvgSigma(data.avg_sigma)
      setUnresolvedItems(data.unresolved_items)
      setPercentFinished(data.percent_finished)
      setTheoreticalMinPairs(data.theoretical_min_pairs)
      
      setIsLoading(false)
      
      // Only fetch OpenAI reasoning if enabled
      if (isOpenAIEnabled) {
        getOpenAIScore(data.image1, data.image2)
      }
    } catch (error) {
      console.error('Error fetching random pair:', error)
      setIsLoading(false)
    }
  }

  const getOpenAIScore = async (image1: string, image2: string) => {
    setOpenAILoading(true)
    try {
      const data = await fetchOpenAIScore(image1, image2)
      setOpenAIScore({
        score: data.score,
        reasoning: data.reasoning
      })
    } catch (error) {
      console.error('Error fetching OpenAI score:', error)
    } finally {
      setOpenAILoading(false)
    }
  }

  const handleFetchOpenAI = () => {
    if (!imagePair) return
    getOpenAIScore(imagePair.image1, imagePair.image2)
  }

  const handleRate = async (winner: string, loser: string) => {
    try {
      await submitRating(winner, loser)
      // Fetch new pair after rating
      getRandomPair()
    } catch (error) {
      console.error('Error submitting rating:', error)
    }
  }

  const handleTie = async () => {
    if (!imagePair) return
    
    try {
      await submitRating(imagePair.image1, imagePair.image2, true)
      // Fetch new pair after rating
      getRandomPair()
    } catch (error) {
      console.error('Error submitting tie:', error)
    }
  }

  useEffect(() => {
    getRandomPair()
  }, [isGridMode])

  return (
    <div className="app-container">
      <Header 
        totalRatings={totalRatings} 
        isGridMode={isGridMode} 
        setIsGridMode={setIsGridMode}
        isOpenAIEnabled={isOpenAIEnabled}
        setIsOpenAIEnabled={setIsOpenAIEnabled}
        avgSigma={avgSigma}
        unresolvedItems={unresolvedItems}
        percentFinished={percentFinished}
        theoreticalMinPairs={theoreticalMinPairs}
      />
      
      <KeyboardShortcutsGuide />
      
      <div className="main-content">
        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading images...</p>
          </div>
        ) : (
          imagePair && (
            <>
              <AIScorePanel 
                aiScore={aiScore} 
                openAIScore={openAIScore} 
                isLoading={openAILoading}
                isOpenAIEnabled={isOpenAIEnabled}
                onFetchOpenAI={handleFetchOpenAI}
              />
              
              <ImageComparison
                imagePair={imagePair}
                onSelectWinner={handleRate}
                onTie={handleTie}
                onFetchOpenAI={handleFetchOpenAI}
              />
            </>
          )
        )}
      </div>
    </div>
  )
}

export default App
