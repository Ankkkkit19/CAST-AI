import { useRef, useEffect, useState, useCallback } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import { GESTURE_DICTIONARY } from '../data/gestures'
import './SignDetector.css'

const BACKEND_URL = 'http://localhost:5001'
const POSE_HOLD_MS = 1000

// MediaPipe hand connections for drawing landmarks
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],         // thumb
  [0,5],[5,6],[6,7],[7,8],         // index
  [0,9],[9,10],[10,11],[11,12],    // middle
  [0,13],[13,14],[14,15],[15,16],  // ring
  [0,17],[17,18],[18,19],[19,20],  // pinky
  [5,9],[9,13],[13,17],            // palm
]

function drawLandmarks(ctx, landmarks, W, H) {
  if (!landmarks || landmarks.length === 0) return
  
  // Calculate bounding box for hand only
  let minX = 1, minY = 1, maxX = 0, maxY = 0
  for (const lm of landmarks) {
    minX = Math.min(minX, lm.x)
    minY = Math.min(minY, lm.y)
    maxX = Math.max(maxX, lm.x)
    maxY = Math.max(maxY, lm.y)
  }
  
  const padX = (maxX - minX) * 0.2
  const padY = (maxY - minY) * 0.2
  minX = Math.max(0, minX - padX)
  minY = Math.max(0, minY - padY)
  maxX = Math.min(1, maxX + padX)
  maxY = Math.min(1, maxY + padY)
  
  const cropX = minX * W
  const cropY = minY * H
  const cropW = (maxX - minX) * W
  const cropH = (maxY - minY) * H
  
  ctx.clearRect(0, 0, W, H)
  ctx.save()
  ctx.rect(cropX, cropY, cropW, cropH)
  ctx.clip()
  
  ctx.lineWidth = 3
  ctx.strokeStyle = 'rgba(124,106,247,0.95)'
  for (const [a, b] of HAND_CONNECTIONS) {
    const lA = landmarks[a], lB = landmarks[b]
    if (!lA || !lB) continue
    ctx.beginPath()
    ctx.moveTo(lA.x * W, lA.y * H)
    ctx.lineTo(lB.x * W, lB.y * H)
    ctx.stroke()
  }
  
  ctx.fillStyle = '#38bdf8'
  for (const lm of landmarks) {
    ctx.beginPath()
    ctx.arc(lm.x * W, lm.y * H, 5.5, 0, 2 * Math.PI)
    ctx.fill()
  }
  
  ctx.strokeStyle = 'rgba(56,189,248,0.4)'
  ctx.lineWidth = 2
  for (const lm of landmarks) {
    ctx.beginPath()
    ctx.arc(lm.x * W, lm.y * H, 8, 0, 2 * Math.PI)
    ctx.stroke()
  }
  
  ctx.restore()
}

export default function SignDetector() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const mpHandsRef = useRef(null)
  const animFrameRef = useRef(null)

  const { settings } = useA11y()

  const [status, setStatus]     = useState('idle')   // idle | loading | active | error
  const [handsDetected, setHandsDetected] = useState(false)
  const [detectedLetter, setDetectedLetter] = useState(null)
  const [registeredLetter, setRegisteredLetter] = useState(null)
  const [sentence, setSentence] = useState('')

  const holdStartRef = useRef(null)
  const lastDetectedRef = useRef(null)
  const confirmedRef = useRef(null)

  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    // A slight delay ensures the cancel is fully processed before speaking
    setTimeout(() => {
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 0.95;
      utt.pitch = 1.1;
      
      // Store reference globally to prevent Chrome garbage collection bug
      window._verbalUtterances = window._verbalUtterances || [];
      window._verbalUtterances.push(utt);
      
      utt.onend = () => {
         window._verbalUtterances = window._verbalUtterances.filter(u => u !== utt);
      };
      
      window.speechSynthesis.speak(utt);
    }, 50);
  }, [])

  const analyzeHand = useCallback(async (landmarks) => {
    if (!landmarks || landmarks.length < 21) {
       holdStartRef.current = null
       lastDetectedRef.current = null
       confirmedRef.current = null
       setDetectedLetter(null)
       return
    }

    let currentDetect = null
    
    // Calculate palm size for relative distances
    const palmSize = Math.hypot(landmarks[5].x - landmarks[17].x, landmarks[5].y - landmarks[17].y);
    
    // General finger extensions
    const indexExt = landmarks[8].y < landmarks[6].y;
    const middleExt = landmarks[12].y < landmarks[10].y;
    const ringExt = landmarks[16].y < landmarks[14].y;
    const pinkyExt = landmarks[20].y < landmarks[18].y;
    
    // Thumb state categorization
    const thumbUp = landmarks[4].y < landmarks[3].y && landmarks[4].y < landmarks[5].y - (palmSize * 0.3);
    const thumbDown = landmarks[4].y > landmarks[3].y && landmarks[4].y > landmarks[5].y + (palmSize * 0.3);
    const thumbOut = Math.abs(landmarks[4].x - landmarks[9].x) > (palmSize * 0.8);

    let thState = "TUCKED";
    if (thumbUp) thState = "UP";
    else if (thumbDown) thState = "DOWN";
    else if (thumbOut) thState = "OUT";

    const code = `${thState}-${indexExt ? '1' : '0'}${middleExt ? '1' : '0'}${ringExt ? '1' : '0'}${pinkyExt ? '1' : '0'}`;

    // Lookup 5-finger pattern in our dictionary
    let match = GESTURE_DICTIONARY.find(g => g.code === code);

    // Override with special distance-based ones that don't purely rely on UP/DOWN
    const thumbIndexDist = Math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y);
    const thumbMiddleDist = Math.hypot(landmarks[4].x - landmarks[12].x, landmarks[4].y - landmarks[12].y);
    
    const isOk = thumbIndexDist < (palmSize * 0.6) && middleExt && ringExt && pinkyExt && !indexExt;
    const isHowMuch = thumbIndexDist < (palmSize * 0.4) && thumbMiddleDist < (palmSize * 0.4) && !ringExt && !pinkyExt;
    const isPinch = thumbIndexDist < (palmSize * 0.4) && !middleExt && !ringExt && !pinkyExt && !thumbOut && !thumbUp && !thumbDown;

    if (isOk) {
      match = GESTURE_DICTIONARY.find(g => g.code === "SPECIAL-OK");
    } else if (isHowMuch) {
      match = GESTURE_DICTIONARY.find(g => g.code === "SPECIAL-MONEY");
    } else if (isPinch) {
      match = GESTURE_DICTIONARY.find(g => g.code === "SPECIAL-PINCH");
    }

    if (match) {
      currentDetect = match.phrase;
    }

    setDetectedLetter(currentDetect)

    if (currentDetect) {
      if (lastDetectedRef.current === currentDetect) {
        if (!holdStartRef.current) {
          holdStartRef.current = Date.now()
        } else if (Date.now() - holdStartRef.current >= POSE_HOLD_MS) {
          if (confirmedRef.current !== currentDetect) {
             setRegisteredLetter(currentDetect)
             setSentence(prev => {
                if (currentDetect.length > 1) {
                  return prev + (prev.length > 0 && !prev.endsWith(' ') ? ' ' : '') + currentDetect + ' ';
                }
                return prev + currentDetect;
             })
             speak(currentDetect)
             confirmedRef.current = currentDetect
          }
        }
      } else {
        lastDetectedRef.current = currentDetect
        holdStartRef.current = Date.now()
      }
    } else {
      lastDetectedRef.current = null
      holdStartRef.current = null
      confirmedRef.current = null
    }

  }, [speak])

  const startCamera = useCallback(async () => {
    setStatus('loading')
    setRegisteredLetter(null)
    setDetectedLetter(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      const waitForMP = (resolve, reject, t = 0) => {
        if (window.Hands) resolve()
        else if (t > 6000) reject(new Error('MediaPipe not loaded'))
        else setTimeout(() => waitForMP(resolve, reject, t + 100), 100)
      }
      await new Promise((res, rej) => waitForMP(res, rej))

      const hands = new window.Hands({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
      })
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.5,
      })

      let frameCount = 0
      hands.onResults(results => {
        const canvas = canvasRef.current
        const video  = videoRef.current
        if (!canvas || !video) return

        const W = canvas.width  = video.videoWidth
        const H = canvas.height = video.videoHeight
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, W, H)

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
          const lms = results.multiHandLandmarks[0]
          setHandsDetected(true)
          drawLandmarks(ctx, lms, W, H)

          frameCount++
          if (frameCount % 6 === 0) analyzeHand(lms)
        } else {
          setHandsDetected(false)
          analyzeHand(null)
        }
      })

      mpHandsRef.current = hands

      const processFrame = async () => {
        if (videoRef.current && videoRef.current.readyState >= 2) {
          await hands.send({ image: videoRef.current })
        }
        animFrameRef.current = requestAnimationFrame(processFrame)
      }
      animFrameRef.current = requestAnimationFrame(processFrame)

      setStatus('active')
    } catch (err) {
      console.error(err)
      setStatus('error')
    }
  }, [analyzeHand])

  const stopCamera = () => {
    cancelAnimationFrame(animFrameRef.current)
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setStatus('idle')
    setHandsDetected(false)
    setDetectedLetter(null)
    setRegisteredLetter(null)
    holdStartRef.current = null
    lastDetectedRef.current = null
    confirmedRef.current = null
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <main id="main-content" className="detector">
      <div className="container">
        {/* Header */}
        <div className="detector__header fade-in-up">
          <div>
            <h1 className="detector__title">🖐️ One-on-One Interpreter</h1>
            <p className="detector__subtitle">
              Show an expression to the camera, and I will read it out loud.
            </p>
          </div>
          <div className="detector__status-badges">
            <span className={`badge ${status === 'active' ? 'badge-green' : 'badge-sky'}`}>
              {status === 'idle' ? '📷 Camera Off' : status === 'loading' ? '⏳ Loading…' : status === 'active' ? '🔴 LIVE' : '❌ Error'}
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', margin: '0 auto', maxWidth: '700px', width: '100%' }}>
          <div className="detector__steps-card glass-card fade-in-up delay-1" style={{ width: '100%', padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', textAlign: 'center' }}>
            
            <p className="detector__pred-label" style={{ fontSize: '1.5rem', marginBottom: '0' }}>Interpreter Result</p>
            
            <div className="detector__step-current glass-card" style={{ width: '100%', padding: status === 'active' ? '1rem' : '3rem', textAlign: 'center', background: 'rgba(255,255,255,0.7)', border: '2px solid rgba(124,106,247,0.2)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', minHeight: '400px', justifyContent: 'center' }}>
              
              {/* Camera Area merged inside the waiting box */}
              {status === 'active' || status === 'loading' ? (
                <div className="detector__video-wrap" style={{ position: 'relative', width: '100%', maxWidth: '500px', aspectRatio: '4/3', borderRadius: '1rem', overflow: 'hidden', backgroundColor: '#000', border: '2px dashed rgba(124,106,247,0.3)', margin: '0 auto' }}>
                  {status === 'loading' && (
                    <div className="detector__placeholder" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(255,255,255,0.8)' }}>
                      <div className="detector__spinner" />
                      <p>Getting ready…</p>
                    </div>
                  )}
                  <video
                    ref={videoRef}
                    id="detector-video"
                    className={`detector__video ${status === 'active' ? 'visible' : ''}`}
                    playsInline muted
                    aria-label="Webcam feed"
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="detector__canvas"
                    aria-hidden="true"
                    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                  />
                  {status === 'active' && handsDetected && (
                    <div className="detector__hand-badge" style={{ position: 'absolute', top: '10px', left: '10px' }}>👋 Hand detected</div>
                  )}
                </div>
              ) : null}

              {status === 'error' && (
                <div className="detector__placeholder">
                  <div style={{ fontSize: '3rem' }}>❌</div>
                  <p>Oops! Camera didn't work. Please allow camera permissions.</p>
                </div>
              )}

              {/* Text Results inside the box */}
              {status === 'active' ? (
                registeredLetter ? (
                  <div style={{ marginTop: '1rem', width: '100%' }}>
                    <h2 style={{ fontSize: '1.5rem', color: '#333' }}>Current Sign:</h2>
                    <div className="detector__step-big-letter" style={{ fontSize: registeredLetter?.length > 1 ? '2rem' : '3rem', margin: '0.5rem 0', color: '#7c6af7', textShadow: '1px 1px 3px rgba(0,0,0,0.1)' }}>
                      {registeredLetter}
                    </div>
                  </div>
                ) : (
                  <div style={{ marginTop: '1rem', width: '100%' }}>
                    <h2 style={{ fontSize: '1.5rem', color: '#666' }}>Waiting for sign...</h2>
                    {detectedLetter ? (
                      <p style={{ fontSize: '1.2rem', color: '#7c6af7', marginTop: '1rem' }}>I see {detectedLetter}, hold steady...</p>
                    ) : (
                      <p style={{ fontSize: '1rem', color: '#999', marginTop: '1rem' }}>Show your hand to the camera</p>
                    )}
                  </div>
                )
              ) : (
                status === 'idle' && (
                  <>
                    <div className="detector__step-big-letter" style={{ fontSize: '5rem', margin: '1rem 0', opacity: 0.2 }}>?</div>
                    <h2 style={{ fontSize: '2rem', color: '#666' }}>Waiting for camera...</h2>
                    <p style={{ fontSize: '1.2rem', color: '#999' }}>Click below to turn on the camera</p>
                  </>
                )
              )}

              {/* Sentence Display & Controls */}
              <div style={{ marginTop: '1.5rem', width: '100%', borderTop: '1px solid rgba(0,0,0,0.1)', paddingTop: '1.5rem' }}>
                <h3 style={{ fontSize: '1.2rem', color: '#555', marginBottom: '0.5rem' }}>Formed Sentence:</h3>
                <div style={{ minHeight: '3rem', padding: '0.5rem 1rem', background: '#f8f9fa', borderRadius: '0.5rem', border: '1px solid #dee2e6', fontSize: '1.5rem', color: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', wordBreak: 'break-word' }}>
                  {sentence || <span style={{ color: '#aaa', fontSize: '1rem' }}>No words formed yet...</span>}
                </div>
                
                <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginTop: '1rem', flexWrap: 'wrap' }}>
                  <button className="btn" style={{ background: '#e2e8f0', color: '#334155', padding: '0.5rem 1rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer' }} onClick={() => setSentence(prev => prev + ' ')}>␣ Space</button>
                  <button className="btn" style={{ background: '#e2e8f0', color: '#334155', padding: '0.5rem 1rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer' }} onClick={() => setSentence(prev => prev.slice(0, -1))}>⌫ Backspace</button>
                  <button className="btn" style={{ background: '#f87171', color: 'white', padding: '0.5rem 1rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer' }} onClick={() => {
                    setSentence('');
                    setRegisteredLetter(null);
                    setDetectedLetter(null);
                    confirmedRef.current = null;
                    lastDetectedRef.current = null;
                    holdStartRef.current = null;
                  }}>🗑 Clear</button>
                  <button className="btn" style={{ background: '#10b981', color: 'white', padding: '0.5rem 1.5rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer', fontWeight: 'bold' }} onClick={() => { if(sentence.trim()) speak(sentence) }}>🔊 Speak Sentence</button>
                </div>
              </div>
            </div>

            <div className="detector__controls" style={{ display: 'flex', justifyContent: 'center', width: '100%', marginTop: '1rem' }}>
              {status === 'idle' || status === 'error' ? (
                <button className="btn btn-primary" style={{ fontSize: '1.2rem', padding: '0.8rem 2rem' }} onClick={startCamera}>
                  🎬 Turn On Camera
                </button>
              ) : (
                <button className="btn btn-danger" style={{ fontSize: '1.2rem', padding: '0.8rem 2rem' }} onClick={stopCamera}>
                  ⏹ Turn Off Camera
                </button>
              )}
            </div>
            
            <div className="detector__tips glass-card" style={{ width: '100%', textAlign: 'left', marginTop: '1rem' }}>
              <p className="detector__pred-label">🎯 How It Works</p>
              <ul className="detector__tips-list" style={{ fontSize: '1rem', lineHeight: '1.6', margin: 0 }}>
                <li>✅ <strong>Turn on the camera</strong> to start.</li>
                <li>🖐️ <strong>Show an expression</strong> to the camera clearly and hold it to register.</li>
                <li>🔤 <strong>Form full thoughts</strong> phrase by phrase.</li>
                <li>🔊 Use the controls to manage your sentence and <strong>speak it out loud!</strong></li>
              </ul>
            </div>

            <div className="detector__dictionary glass-card" style={{ width: '100%', textAlign: 'left', marginTop: '1rem' }}>
              <p className="detector__pred-label">📚 Supported Sentences Reference ({GESTURE_DICTIONARY.length} mapped)</p>
              <p style={{ color: '#666', marginBottom: '1rem', fontSize: '0.95rem' }}>The interpreter now maps <strong>dozens</strong> of static finger configurations. Play around with different hand shapes to discover them in real-time!</p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.8rem', fontSize: '0.9rem', color: '#334155', maxHeight: '500px', overflowY: 'auto', padding: '0.5rem' }}>
                {GESTURE_DICTIONARY.map((g, i) => (
                   <div key={i} style={{ background: 'rgba(255,255,255,0.7)', padding: '0.8rem', borderRadius: '0.5rem', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                      <span style={{ fontSize: '1.2rem', marginRight: '0.5rem' }}>{g.emoji}</span> <strong>{g.phrase}</strong>
                   </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </div>
    </main>
  )
}
