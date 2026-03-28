import { useState, useEffect, useRef } from 'react'
import { A11yText } from '../context/AccessibilityContext'
import './VideoTranscriber.css'

export default function VideoTranscriber({ setPage }) {
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [liveTranscript, setLiveTranscript] = useState('')
  const [error, setError] = useState(null)
  const videoRef = useRef(null)
  const recognitionRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => {
    // Check for browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setError("Your browser does not support the Web Speech API. Please use Chrome or Edge.")
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onresult = (event) => {
      let finalTrans = ''
      let interimTrans = ''
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTrans += event.results[i][0].transcript + ' '
        } else {
          interimTrans += event.results[i][0].transcript
        }
      }
      if (finalTrans) {
        setTranscript(prev => prev + finalTrans)
      }
      setLiveTranscript(interimTrans)
    }

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      if (event.error !== 'no-speech') {
        setError("Error occurred in recognition: " + event.error)
      }
    }

    recognition.onend = () => {
      // If we are still supposed to be recording, restart it to keep listening
      if (isRecording) {
        try {
          recognition.start()
        } catch (e) {
          console.error("Restart error:", e)
        }
      }
    }

    recognitionRef.current = recognition
  }, [isRecording])

  const startRecording = async () => {
    try {
      if (error && document.location.protocol !== 'https:' && document.location.hostname !== 'localhost') {
        setError("Speech Recognition requires HTTPS.")
        return
      }

      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      if (recognitionRef.current) {
        recognitionRef.current.start()
        setIsRecording(true)
        setError(null)
      }
    } catch (err) {
      console.error(err)
      setError("Please allow camera and microphone access to use this feature.")
    }
  }

  const stopRecording = () => {
    setIsRecording(false)
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="video-transcriber-page fade-in-up">
      <div className="container">
        <header className="transcriber-header">
          <button className="btn btn-ghost" onClick={() => setPage('dashboard')}>
            ← Back to Dashboard
          </button>
          <A11yText as="h1">Live Video Transcriber</A11yText>
          <p>Real-time subtitles for whatever you say to the camera.</p>
        </header>

        {error ? (
          <div className="transcriber-error">
            <p>⚠️ {error}</p>
          </div>
        ) : null}

        <div className="transcriber-container glass-card">
          <div className="video-wrapper">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className={`live-video ${!isRecording ? 'hidden-vid' : ''}`}
            />
            {!isRecording && (
              <div className="video-placeholder">
                <span className="placeholder-icon">📷</span>
                <p>Camera is off</p>
              </div>
            )}
            
            {/* Live Subtitle Overlay */}
            {(liveTranscript || transcript) && isRecording && (
              <div className="live-captions-overlay">
                <span className="caption-text">
                  <span className="final-text">{transcript.split(' ').slice(-15).join(' ')}</span>
                  {' '}
                  <span className="interim-text">{liveTranscript}</span>
                </span>
              </div>
            )}
          </div>

          <div className="transcriber-controls">
            {!isRecording ? (
              <button className="btn btn-primary lg" onClick={startRecording}>
                ▶ Start Transcript Video
              </button>
            ) : (
              <button className="btn btn-danger lg" onClick={stopRecording}>
                ⏹ Stop Recording
              </button>
            )}
          </div>

          <div className="transcript-history-box">
            <h3>Full Transcript History</h3>
            <div className="transcript-content">
              {transcript.length > 0 ? (
                <p>{transcript} <span className="live-typing">{liveTranscript}</span></p>
              ) : (
                <p className="no-text-yet">No speech detected yet. Start the video and say something...</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
