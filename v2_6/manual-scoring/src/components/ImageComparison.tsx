import React, { useState, useEffect, useRef } from 'react'
import { ImagePair } from '../types'
import { getImageUrl } from '../utils/api'

interface ImageComparisonProps {
  imagePair: ImagePair
  onSelectWinner: (winner: string, loser: string) => void
  onTie: () => void
  onFetchOpenAI: () => void
}

const ImageComparison: React.FC<ImageComparisonProps> = ({ 
  imagePair,
  onSelectWinner,
  onTie,
  onFetchOpenAI
}) => {
  const [hoveredImage, setHoveredImage] = useState<string | null>(null)
  const [loading, setLoading] = useState<{[key: string]: boolean}>({
    [imagePair.image1]: true,
    [imagePair.image2]: true
  })
  const [activeKeyboardShortcut, setActiveKeyboardShortcut] = useState<string | null>(null)
  const image1Ref = useRef<HTMLImageElement>(null)
  const image2Ref = useRef<HTMLImageElement>(null)

  const handleImageLoad = (imageId: string) => {
    setLoading(prev => ({
      ...prev,
      [imageId]: false
    }))
  }

  // Handle click events for selecting winners
  const handleWinnerSelection = (winner: string, loser: string) => {
    onSelectWinner(winner, loser)
  }

  // Check for cached images that may have completed loading before we attached onLoad
  useEffect(() => {
    const checkImagesLoaded = () => {
      if (image1Ref.current?.complete) {
        handleImageLoad(imagePair.image1)
      }
      
      if (image2Ref.current?.complete) {
        handleImageLoad(imagePair.image2)
      }
    }
    
    // Run immediately and also after a short delay to catch edge cases
    checkImagesLoaded()
    const timer = setTimeout(checkImagesLoaded, 100)
    
    return () => clearTimeout(timer)
  }, [imagePair])

  // Listen for keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowLeft':
          setActiveKeyboardShortcut('left')
          setTimeout(() => setActiveKeyboardShortcut(null), 300)
          onSelectWinner(imagePair.image1, imagePair.image2)
          break;
        case 'ArrowRight':
          setActiveKeyboardShortcut('right')
          setTimeout(() => setActiveKeyboardShortcut(null), 300)
          onSelectWinner(imagePair.image2, imagePair.image1)
          break;
        case ' ':
        case 'Enter':
          setActiveKeyboardShortcut('tie')
          setTimeout(() => setActiveKeyboardShortcut(null), 300)
          onTie()
          break;
        case 'f':
        case 'F':
          onFetchOpenAI()
          break;
        default:
          break;
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [imagePair, onSelectWinner, onTie, onFetchOpenAI])

  // Reset loading state when images change
  useEffect(() => {
    setLoading({
      [imagePair.image1]: true,
      [imagePair.image2]: true
    })
  }, [imagePair])

  return (
    <div className="image-comparison-container">
      <div className="image-comparison">
        <div 
          className={`image-card ${hoveredImage === imagePair.image1 ? 'hovered' : ''} ${activeKeyboardShortcut === 'left' ? 'keyboard-active' : ''}`}
          onMouseEnter={() => setHoveredImage(imagePair.image1)}
          onMouseLeave={() => setHoveredImage(null)}
          onClick={() => handleWinnerSelection(imagePair.image1, imagePair.image2)}
        >
          {loading[imagePair.image1] && (
            <div className="image-loading-indicator">
              <div className="spinner"></div>
            </div>
          )}
          <img 
            ref={image1Ref}
            src={getImageUrl(imagePair.image1)}
            alt="Image 1"
            onLoad={() => handleImageLoad(imagePair.image1)}
          />
          <button 
            className="winner-selector"
          >
            <span>Select as Winner</span>
          </button>
        </div>

        <div 
          className={`image-card ${hoveredImage === imagePair.image2 ? 'hovered' : ''} ${activeKeyboardShortcut === 'right' ? 'keyboard-active' : ''}`}
          onMouseEnter={() => setHoveredImage(imagePair.image2)}
          onMouseLeave={() => setHoveredImage(null)}
          onClick={() => handleWinnerSelection(imagePair.image2, imagePair.image1)}
        >
          {loading[imagePair.image2] && (
            <div className="image-loading-indicator">
              <div className="spinner"></div>
            </div>
          )}
          <img 
            ref={image2Ref}
            src={getImageUrl(imagePair.image2)}
            alt="Image 2"
            onLoad={() => handleImageLoad(imagePair.image2)}
          />
          <button 
            className="winner-selector"
          >
            <span>Select as Winner</span>
          </button>
        </div>
      </div>

      <div className="comparison-controls">
        <button 
          className={`tie-button ${activeKeyboardShortcut === 'tie' ? 'keyboard-active' : ''}`}
          onClick={onTie}
        >
          It's a Tie
        </button>
      </div>
    </div>
  )
}

export default ImageComparison