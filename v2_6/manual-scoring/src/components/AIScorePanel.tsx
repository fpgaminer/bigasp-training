import React, { useState } from 'react'
import { OpenAIScoreResult } from '../types'
import { getTextToSpeech } from '../utils/api'

interface AIScorePanelProps {
  aiScore: number
  openAIScore: OpenAIScoreResult | null
  isLoading: boolean
  isOpenAIEnabled: boolean
  onFetchOpenAI: () => void
}

const AIScorePanel: React.FC<AIScorePanelProps> = ({ 
  aiScore, 
  openAIScore, 
  isLoading, 
  isOpenAIEnabled, 
  onFetchOpenAI 
}) => {
  const [isPlayingAudio, setIsPlayingAudio] = useState<boolean>(false)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  
  // Convert AI score to a percentage
  const scorePercent = Math.round(aiScore * 100)
  const formattedScore = `${scorePercent}%`
  
  // Determine which image OpenAI preferred (if available)
  const openAIPreferred = openAIScore 
    ? openAIScore.score < 0.5 
      ? "Left" 
      : "Right"
    : null

  const handlePlayAudio = async () => {
    if (!openAIScore?.reasoning) return;
    
    if (audioUrl) {
      // If we already have the audio URL, just play it
      const audio = new Audio(audioUrl);
      setIsPlayingAudio(true);
      audio.play();
      audio.onended = () => setIsPlayingAudio(false);
      return;
    }
    
    try {
      setIsPlayingAudio(true);
      
      // Prepare the text for speech synthesis
      const speechText = `The winner is ${openAIPreferred} image. ${openAIScore.reasoning}`;
      
      // Get the audio data using our utility function
      const audioData = await getTextToSpeech(speechText);
      
      // Create an audio element and play it
      setAudioUrl(audioData);
      const audio = new Audio(audioData);
      audio.play();
      audio.onended = () => setIsPlayingAudio(false);
    } catch (error) {
      console.error('Error fetching audio:', error);
      setIsPlayingAudio(false);
    }
  };

  return (
    <div className="ai-score-panel">
      <div className="score-section">
        <h3>Quality Classifier</h3>
        <div className="score-bar-container">
          <div 
            className="score-bar" 
            style={{ width: formattedScore }}
            title={`${formattedScore} chance that right image is better`}
          ></div>
        </div>
        <div className="score-labels">
          <span>Left</span>
          <span className="score-value">{formattedScore}</span>
          <span>Right</span>
        </div>
      </div>

      <div className="openai-section">
        <div className="openai-header">
          <h3>
            AI Reasoning
            {isLoading && <span className="pulse-dot"></span>}
            {openAIScore && openAIPreferred && (
              <span className="openai-preferred-inline"> • Prefers: {openAIPreferred} Image</span>
            )}
          </h3>
          
          <div className="openai-controls">
            {!isOpenAIEnabled && !openAIScore && !isLoading && (
              <button 
                className="fetch-button"
                onClick={onFetchOpenAI}
                disabled={isLoading}
                title="Get AI reasoning for this image pair"
              >
                Fetch
              </button>
            )}
            
            {openAIScore && openAIScore.reasoning && (
              <button 
                className={`audio-button ${isPlayingAudio ? 'playing' : ''}`}
                onClick={handlePlayAudio}
                disabled={isLoading || isPlayingAudio || !openAIScore}
                title="Listen to AI reasoning"
              >
                {isPlayingAudio ? '▶️' : '🔊'}
              </button>
            )}
          </div>
        </div>
        
        {openAIScore && (
          <>            
            <div className="openai-reasoning">
              <p>{openAIScore.reasoning}</p>
            </div>
          </>
        )}
        
        {!openAIScore && !isLoading && (
          <div className="no-data">
            {isOpenAIEnabled ? 'AI reasoning not available' : 'AI reasoning not fetched'}
          </div>
        )}
        
        {isLoading && (
          <div className="loading-container">
            <p>Getting AI opinion...</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default AIScorePanel