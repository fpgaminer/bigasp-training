import React, { useState, useEffect } from 'react'

const KeyboardShortcutsGuide: React.FC = () => {
  const [isVisible, setIsVisible] = useState(true)
  
  useEffect(() => {
    // Hide the guide after 10 seconds
    const timer = setTimeout(() => {
      setIsVisible(false)
    }, 10000)
    
    return () => clearTimeout(timer)
  }, [])
  
  if (!isVisible) return null
  
  return (
    <div className="keyboard-shortcuts-guide">
      <div className="guide-content">
        <h3>Keyboard Shortcuts</h3>
        <div className="shortcut-items">
          <div className="shortcut-item">
            <kbd>←</kbd>
            <span>Select left image as winner</span>
          </div>
          <div className="shortcut-item">
            <kbd>→</kbd>
            <span>Select right image as winner</span>
          </div>
          <div className="shortcut-item">
            <kbd>Space</kbd>
            <span>It's a tie</span>
          </div>
          <div className="shortcut-item">
            <kbd>F</kbd>
            <span>Fetch AI reasoning</span>
          </div>
        </div>
        <button 
          className="close-guide-button"
          onClick={() => setIsVisible(false)}
        >
          Got it
        </button>
      </div>
    </div>
  )
}

export default KeyboardShortcutsGuide