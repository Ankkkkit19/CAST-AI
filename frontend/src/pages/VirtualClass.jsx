import { useRef, useEffect, useState, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { GESTURE_DICTIONARY } from '../data/gestures';
import {
  broadcastSign,
  subscribeToClassSession,
  startClassSession,
  endClassSession,
} from '../services/virtualClassService';

const POSE_HOLD_MS = 1200;
const ROOM_NAME = 'VardaanInclusiveClassroom_101';

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

function drawLandmarks(ctx, landmarks, W, H) {
  if (!landmarks || landmarks.length === 0) return;
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = 'rgba(124,106,247,0.9)';
  for (const [a, b] of HAND_CONNECTIONS) {
    const lA = landmarks[a], lB = landmarks[b];
    if (!lA || !lB) continue;
    ctx.beginPath();
    ctx.moveTo(lA.x * W, lA.y * H);
    ctx.lineTo(lB.x * W, lB.y * H);
    ctx.stroke();
  }
  ctx.fillStyle = '#38bdf8';
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * W, lm.y * H, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

export default function VirtualClass({ onBack, setPage }) {
  const { currentUser, userRole } = useAuth();
  const isTeacher = userRole === 'teacher';

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mpHandsRef = useRef(null);
  const animFrameRef = useRef(null);
  const holdStartRef = useRef(null);
  const lastDetectedRef = useRef(null);
  const confirmedRef = useRef(null);
  const joinedAtRef = useRef(Date.now());

  const [camStatus, setCamStatus] = useState('idle'); // idle | loading | active | error
  const [detectedSign, setDetectedSign] = useState(null);
  const [confirmedSign, setConfirmedSign] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [lastReceivedSign, setLastReceivedSign] = useState(null);
  const [lastSignEmoji, setLastSignEmoji] = useState('🤟');
  const [holdProgress, setHoldProgress] = useState(0);
  const [broadcastHistory, setBroadcastHistory] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // ── Text-to-speech ─────────────────────────────────────────────────────────
  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    setIsSpeaking(true);
    setTimeout(() => {
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 0.92;
      utt.pitch = 1.05;
      utt.onend = () => setIsSpeaking(false);
      utt.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utt);
    }, 60);
  }, []);

  // ── Firebase subscription ───────────────────────────────────────────────────
  useEffect(() => {
    const unsub = subscribeToClassSession((data) => {
      setSessionActive(!!data.active);

      const signTime = data.lastSignAt?.toMillis?.() ?? 0;
      const isNew = signTime > joinedAtRef.current;

      if (data.lastSign && isNew && data.lastSign !== lastReceivedSign) {
        const entry = {
          phrase: data.lastSign,
          emoji: data.lastSignEmoji || '🤟',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        };
        setBroadcastHistory(prev => [entry, ...prev].slice(0, 8));
        setLastReceivedSign(data.lastSign);
        setLastSignEmoji(data.lastSignEmoji || '🤟');
        if (!isTeacher) speak(data.lastSign);
      }
    });
    return () => unsub();
  }, [isTeacher, lastReceivedSign, speak]);

  // ── Gesture analysis ────────────────────────────────────────────────────────
  const analyzeHand = useCallback((landmarks) => {
    if (!landmarks || landmarks.length < 21) {
      holdStartRef.current = null;
      lastDetectedRef.current = null;
      confirmedRef.current = null;
      setDetectedSign(null);
      setHoldProgress(0);
      return;
    }

    const palmSize = Math.hypot(landmarks[5].x - landmarks[17].x, landmarks[5].y - landmarks[17].y);
    const indexExt = landmarks[8].y < landmarks[6].y;
    const middleExt = landmarks[12].y < landmarks[10].y;
    const ringExt   = landmarks[16].y < landmarks[14].y;
    const pinkyExt  = landmarks[20].y < landmarks[18].y;

    const thumbUp   = landmarks[4].y < landmarks[3].y && landmarks[4].y < landmarks[5].y - palmSize * 0.3;
    const thumbDown = landmarks[4].y > landmarks[3].y && landmarks[4].y > landmarks[5].y + palmSize * 0.3;
    const thumbOut  = Math.abs(landmarks[4].x - landmarks[9].x) > palmSize * 0.8;

    let thState = 'TUCKED';
    if (thumbUp)   thState = 'UP';
    else if (thumbDown) thState = 'DOWN';
    else if (thumbOut)  thState = 'OUT';

    const code = `${thState}-${indexExt?'1':'0'}${middleExt?'1':'0'}${ringExt?'1':'0'}${pinkyExt?'1':'0'}`;
    let match = GESTURE_DICTIONARY.find(g => g.code === code);

    const thumbIndexDist = Math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y);
    if (thumbIndexDist < palmSize * 0.6 && middleExt && ringExt && pinkyExt && !indexExt) {
      match = GESTURE_DICTIONARY.find(g => g.code === 'SPECIAL-OK');
    }

    const currentDetect = match ? match.phrase : null;
    const currentEmoji  = match ? match.emoji  : '🤟';
    setDetectedSign(currentDetect);

    if (currentDetect && isTeacher) {
      if (lastDetectedRef.current === currentDetect) {
        if (!holdStartRef.current) holdStartRef.current = Date.now();
        const elapsed = Date.now() - holdStartRef.current;
        setHoldProgress(Math.min(100, (elapsed / POSE_HOLD_MS) * 100));

        if (elapsed >= POSE_HOLD_MS && confirmedRef.current !== currentDetect) {
          confirmedRef.current = currentDetect;
          setConfirmedSign(currentDetect);
          setHoldProgress(0);
          broadcastSign(currentDetect, currentEmoji);
          const entry = {
            phrase: currentDetect,
            emoji: currentEmoji,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          };
          setBroadcastHistory(prev => [entry, ...prev].slice(0, 8));
        }
      } else {
        lastDetectedRef.current = currentDetect;
        holdStartRef.current = Date.now();
        confirmedRef.current = null;
        setHoldProgress(0);
      }
    } else {
      lastDetectedRef.current = null;
      holdStartRef.current = null;
      confirmedRef.current = null;
      setHoldProgress(0);
    }
  }, [isTeacher]);

  // ── Camera ──────────────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCamStatus('loading');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      await new Promise((res) => {
        const wait = () => window.Hands ? res() : setTimeout(wait, 100);
        wait();
      });

      const hands = new window.Hands({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
      });
      hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.65, minTrackingConfidence: 0.5 });

      let frameCount = 0;
      hands.onResults(results => {
        const canvas = canvasRef.current;
        const video  = videoRef.current;
        if (!canvas || !video) return;
        const W = canvas.width  = video.videoWidth;
        const H = canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, W, H);
        if (results.multiHandLandmarks?.length > 0) {
          drawLandmarks(ctx, results.multiHandLandmarks[0], W, H);
          frameCount++;
          if (frameCount % 5 === 0) analyzeHand(results.multiHandLandmarks[0]);
        } else {
          analyzeHand(null);
        }
      });

      mpHandsRef.current = hands;
      const processFrame = async () => {
        if (videoRef.current?.readyState >= 2) await hands.send({ image: videoRef.current });
        animFrameRef.current = requestAnimationFrame(processFrame);
      };
      animFrameRef.current = requestAnimationFrame(processFrame);

      setCamStatus('active');
      if (isTeacher) startClassSession(currentUser?.uid || 'teacher');
    } catch (err) {
      console.error(err);
      setCamStatus('error');
    }
  }, [analyzeHand, isTeacher, currentUser]);

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setCamStatus('idle');
    setDetectedSign(null);
    setHoldProgress(0);
    if (isTeacher) endClassSession();
  }, [isTeacher]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  // ── Back handler ────────────────────────────────────────────────────────────
  const handleBack = () => {
    stopCamera();
    if (onBack) onBack();
    else if (setPage) setPage(isTeacher ? 'teacherDashboard' : 'dashboard');
  };

  // ────────────────────────────────────────────────────────────────────────────
  return (
    <div className="vc-root">
      {/* ── Top bar ── */}
      <div className="vc-topbar">
        <div className="vc-topbar-left">
          <button className="vc-back-btn" onClick={handleBack}>
            <span>←</span> Back
          </button>
          <div className="vc-title-wrap">
            <div className="vc-live-dot" />
            <h1 className="vc-title">
              {isTeacher ? '🎓 Teaching Session' : '👩‍🎓 Joining Class'}
            </h1>
            <span className={`vc-session-badge ${sessionActive ? 'active' : 'inactive'}`}>
              {sessionActive ? 'LIVE' : 'WAITING'}
            </span>
          </div>
        </div>
        <div className="vc-room-label">Room: {ROOM_NAME}</div>
      </div>

      {/* ── Main layout ── */}
      <div className="vc-layout">
        {/* ── Video call panel ── */}
        <div className="vc-jitsi-panel">
          <iframe
            title="Jitsi Meeting"
            src={`https://meet.jit.si/${ROOM_NAME}#config.prejoinPageEnabled=false&config.startWithAudioMuted=false&config.startWithVideoMuted=false`}
            allow="camera; microphone; fullscreen; display-capture"
            className="vc-iframe"
          />
        </div>

        {/* ── Communication side panel ── */}
        <div className="vc-comm-panel">

          {isTeacher ? (
            /* ════════════════ TEACHER VIEW ════════════════ */
            <>
              <div className="vc-section-header">
                <span className="vc-section-icon">🤟</span>
                <div>
                  <h2 className="vc-section-title">Sign Broadcaster</h2>
                  <p className="vc-section-sub">Hold a sign for 1.2s to broadcast to all students</p>
                </div>
              </div>

              {/* Camera */}
              <div className="vc-camera-box">
                {camStatus === 'active' ? (
                  <div className="vc-video-wrap">
                    <video ref={videoRef} playsInline muted className="vc-video" />
                    <canvas ref={canvasRef} className="vc-canvas" />
                    {detectedSign && (
                      <div className="vc-overlay-sign">
                        <span className="vc-overlay-text">{detectedSign}</span>
                        <div className="vc-hold-bar">
                          <div className="vc-hold-fill" style={{ width: `${holdProgress}%` }} />
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="vc-camera-placeholder">
                    {camStatus === 'loading' && <div className="vc-spinner" />}
                    {camStatus === 'idle' && <span className="vc-placeholder-icon">📷</span>}
                    {camStatus === 'error' && <span className="vc-placeholder-icon">⚠️</span>}
                    <p>{camStatus === 'loading' ? 'Loading MediaPipe…' : camStatus === 'error' ? 'Camera error' : 'Camera off'}</p>
                  </div>
                )}
              </div>

              <button
                className={`vc-cam-btn ${camStatus === 'active' ? 'stop' : 'start'}`}
                onClick={camStatus === 'active' ? stopCamera : startCamera}
                disabled={camStatus === 'loading'}
              >
                {camStatus === 'active' ? '⏹ Stop Broadcasting' : camStatus === 'loading' ? '⏳ Initializing…' : '🎬 Start Broadcasting'}
              </button>

              {/* Last confirmed broadcast */}
              {confirmedSign && (
                <div className="vc-confirmed-box">
                  <p className="vc-confirmed-label">Last Broadcast</p>
                  <div className="vc-confirmed-text">✅ {confirmedSign}</div>
                </div>
              )}
            </>
          ) : (
            /* ════════════════ STUDENT VIEW ════════════════ */
            <>
              <div className="vc-section-header">
                <span className="vc-section-icon">👂</span>
                <div>
                  <h2 className="vc-section-title">Teacher's Signs</h2>
                  <p className="vc-section-sub">Signs from the teacher appear here and are spoken aloud automatically</p>
                </div>
              </div>

              {/* Live sign display */}
              <div className={`vc-student-sign-box ${lastReceivedSign ? 'has-sign' : ''}`}>
                {!sessionActive ? (
                  <div className="vc-waiting">
                    <div className="vc-waiting-pulse">⏳</div>
                    <p>Waiting for teacher to start class…</p>
                  </div>
                ) : lastReceivedSign ? (
                  <div className="vc-sign-display">
                    <div className="vc-sign-emoji">{lastSignEmoji}</div>
                    <div className="vc-sign-phrase">{lastReceivedSign}</div>
                    {isSpeaking && (
                      <div className="vc-speaking-indicator">
                        <span className="vc-sound-bar" /><span className="vc-sound-bar" /><span className="vc-sound-bar" /><span className="vc-sound-bar" />
                        <span className="vc-speaking-text">Speaking…</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="vc-waiting">
                    <div className="vc-waiting-pulse">🎙️</div>
                    <p>Session is live. Awaiting first sign…</p>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ── Shared: Broadcast history ── */}
          {broadcastHistory.length > 0 && (
            <div className="vc-history">
              <p className="vc-history-label">📜 Recent Signs</p>
              <div className="vc-history-list">
                {broadcastHistory.map((h, i) => (
                  <div key={i} className="vc-history-item" style={{ opacity: 1 - i * 0.1 }}>
                    <span className="vc-hist-emoji">{h.emoji}</span>
                    <span className="vc-hist-phrase">{h.phrase}</span>
                    <span className="vc-hist-time">{h.time}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        /* ── Root & Layout ── */
        .vc-root {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: #0c0e1a;
          color: #e2e8f0;
          font-family: 'Inter', 'Segoe UI', sans-serif;
          overflow: hidden;
        }

        /* ── Topbar ── */
        .vc-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.75rem 1.5rem;
          background: rgba(255,255,255,0.04);
          border-bottom: 1px solid rgba(255,255,255,0.08);
          backdrop-filter: blur(10px);
          flex-shrink: 0;
        }
        .vc-topbar-left { display: flex; align-items: center; gap: 1rem; }
        .vc-back-btn {
          display: flex; align-items: center; gap: 0.4rem;
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.12);
          color: #94a3b8; border-radius: 8px;
          padding: 0.4rem 0.9rem; cursor: pointer;
          font-size: 0.85rem; transition: all 0.2s;
        }
        .vc-back-btn:hover { background: rgba(255,255,255,0.12); color: #e2e8f0; }
        .vc-title-wrap { display: flex; align-items: center; gap: 0.6rem; }
        .vc-live-dot {
          width: 8px; height: 8px;
          border-radius: 50%; background: #22c55e;
          box-shadow: 0 0 8px #22c55e;
          animation: vcPulse 1.5s infinite;
        }
        @keyframes vcPulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        .vc-title { font-size: 1.05rem; font-weight: 600; margin: 0; letter-spacing: 0.01em; }
        .vc-session-badge {
          font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
          padding: 0.2rem 0.5rem; border-radius: 20px;
        }
        .vc-session-badge.active { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
        .vc-session-badge.inactive { background: rgba(148,163,184,0.1); color: #64748b; border: 1px solid rgba(148,163,184,0.2); }
        .vc-room-label { font-size: 0.78rem; color: #475569; font-family: monospace; }

        /* ── Main Layout ── */
        .vc-layout {
          display: grid;
          grid-template-columns: 1fr 420px;
          gap: 0;
          flex: 1;
          min-height: 0;
        }

        /* ── Jitsi ── */
        .vc-jitsi-panel {
          height: 100%;
          background: #000;
          border-right: 1px solid rgba(255,255,255,0.06);
        }
        .vc-iframe { width: 100%; height: 100%; border: none; display: block; }

        /* ── Comm Panel ── */
        .vc-comm-panel {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          padding: 1.25rem;
          overflow-y: auto;
          background: #10121f;
        }

        .vc-section-header {
          display: flex; gap: 0.85rem; align-items: flex-start;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }
        .vc-section-icon { font-size: 1.6rem; line-height: 1; }
        .vc-section-title { font-size: 1rem; font-weight: 700; margin: 0 0 0.15rem; }
        .vc-section-sub { font-size: 0.78rem; color: #64748b; margin: 0; line-height: 1.4; }

        /* ── Camera box (Teacher) ── */
        .vc-camera-box {
          border-radius: 12px; overflow: hidden;
          background: #0a0c18;
          border: 1px solid rgba(124,106,247,0.2);
          min-height: 200px;
        }
        .vc-video-wrap { position: relative; width: 100%; }
        .vc-video { width: 100%; display: block; }
        .vc-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

        .vc-overlay-sign {
          position: absolute; bottom: 0; left: 0; right: 0;
          background: linear-gradient(transparent, rgba(0,0,0,0.85));
          padding: 1rem 0.75rem 0.75rem;
        }
        .vc-overlay-text {
          font-size: 0.85rem; font-weight: 600; color: #c4b5fd;
          display: block; margin-bottom: 0.4rem;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .vc-hold-bar {
          height: 4px; background: rgba(255,255,255,0.15);
          border-radius: 2px; overflow: hidden;
        }
        .vc-hold-fill {
          height: 100%;
          background: linear-gradient(90deg, #7c6af7, #38bdf8);
          transition: width 0.1s linear;
          border-radius: 2px;
        }

        .vc-camera-placeholder {
          min-height: 200px; display: flex; flex-direction: column;
          align-items: center; justify-content: center; gap: 0.5rem;
          color: #475569;
        }
        .vc-placeholder-icon { font-size: 2.5rem; }
        .vc-spinner {
          width: 36px; height: 36px;
          border: 3px solid rgba(124,106,247,0.3);
          border-top-color: #7c6af7;
          border-radius: 50%; animation: vcSpin 0.9s linear infinite;
        }
        @keyframes vcSpin { to { transform: rotate(360deg); } }

        /* ── Cam button ── */
        .vc-cam-btn {
          width: 100%; padding: 0.75rem 1rem;
          border: none; border-radius: 10px;
          font-weight: 600; font-size: 0.9rem;
          cursor: pointer; transition: all 0.2s;
          letter-spacing: 0.02em;
        }
        .vc-cam-btn.start {
          background: linear-gradient(135deg, #7c6af7, #38bdf8);
          color: #fff; box-shadow: 0 4px 20px rgba(124,106,247,0.35);
        }
        .vc-cam-btn.start:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(124,106,247,0.5); }
        .vc-cam-btn.stop { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
        .vc-cam-btn.stop:hover { background: rgba(239,68,68,0.25); }
        .vc-cam-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        /* ── Confirmed sign ── */
        .vc-confirmed-box {
          background: rgba(34,197,94,0.08);
          border: 1px solid rgba(34,197,94,0.25);
          border-radius: 10px; padding: 0.75rem 1rem;
        }
        .vc-confirmed-label { font-size: 0.72rem; color: #4ade80; margin: 0 0 0.3rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
        .vc-confirmed-text { font-size: 0.9rem; font-weight: 600; color: #a7f3d0; }

        /* ── Student sign display ── */
        .vc-student-sign-box {
          flex: 1; min-height: 200px;
          border-radius: 14px;
          border: 1px solid rgba(56,189,248,0.2);
          background: rgba(56,189,248,0.04);
          display: flex; align-items: center; justify-content: center;
          text-align: center; overflow: hidden;
          transition: border-color 0.4s, background 0.4s;
        }
        .vc-student-sign-box.has-sign {
          border-color: rgba(56,189,248,0.5);
          background: rgba(56,189,248,0.08);
          animation: vcSignGlow 0.6s ease;
        }
        @keyframes vcSignGlow {
          0% { box-shadow: 0 0 0px rgba(56,189,248,0); }
          50% { box-shadow: 0 0 30px rgba(56,189,248,0.4); }
          100% { box-shadow: 0 0 0px rgba(56,189,248,0); }
        }
        .vc-waiting { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; color: #475569; padding: 2rem; }
        .vc-waiting-pulse { font-size: 2.5rem; animation: vcPulse 2s infinite; }
        .vc-sign-display { padding: 1.5rem; }
        .vc-sign-emoji {
          font-size: 3.5rem; line-height: 1;
          display: block; margin-bottom: 0.6rem;
          animation: vcBounceIn 0.4s cubic-bezier(0.22, 1, 0.36, 1);
        }
        @keyframes vcBounceIn {
          0% { transform: scale(0.5); opacity: 0; }
          100% { transform: scale(1); opacity: 1; }
        }
        .vc-sign-phrase {
          font-size: 1.15rem; font-weight: 700;
          color: #7dd3fc; line-height: 1.35;
          animation: vcSlideUp 0.35s ease;
        }
        @keyframes vcSlideUp {
          from { transform: translateY(10px); opacity: 0; }
          to   { transform: translateY(0);  opacity: 1; }
        }

        /* ── Speaking indicator ── */
        .vc-speaking-indicator {
          display: flex; align-items: center; justify-content: center;
          gap: 0.3rem; margin-top: 0.75rem;
        }
        .vc-sound-bar {
          width: 4px; border-radius: 2px; background: #38bdf8;
          animation: vcSoundWave 0.7s ease-in-out infinite alternate;
        }
        .vc-sound-bar:nth-child(1) { height: 8px; animation-delay: 0s; }
        .vc-sound-bar:nth-child(2) { height: 16px; animation-delay: 0.15s; }
        .vc-sound-bar:nth-child(3) { height: 12px; animation-delay: 0.3s; }
        .vc-sound-bar:nth-child(4) { height: 20px; animation-delay: 0.45s; }
        @keyframes vcSoundWave {
          from { transform: scaleY(0.4); opacity: 0.5; }
          to   { transform: scaleY(1);   opacity: 1; }
        }
        .vc-speaking-text { font-size: 0.75rem; color: #38bdf8; font-weight: 600; }

        /* ── History ── */
        .vc-history {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 12px; padding: 0.85rem;
        }
        .vc-history-label { font-size: 0.72rem; color: #475569; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; margin: 0 0 0.6rem; }
        .vc-history-list { display: flex; flex-direction: column; gap: 0.35rem; }
        .vc-history-item {
          display: flex; align-items: center; gap: 0.5rem;
          padding: 0.4rem 0.5rem; border-radius: 7px;
          background: rgba(255,255,255,0.03); font-size: 0.82rem;
          transition: background 0.2s;
        }
        .vc-history-item:hover { background: rgba(255,255,255,0.06); }
        .vc-hist-emoji { font-size: 1rem; flex-shrink: 0; }
        .vc-hist-phrase { flex: 1; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .vc-hist-time { font-size: 0.7rem; color: #475569; flex-shrink: 0; font-family: monospace; }

        /* ── Responsive ── */
        @media (max-width: 900px) {
          .vc-layout { grid-template-columns: 1fr; grid-template-rows: 50vh 1fr; overflow-y: auto; }
          .vc-jitsi-panel { height: 50vh; border-right: none; border-bottom: 1px solid rgba(255,255,255,0.06); }
          .vc-root { height: auto; overflow: auto; }
        }
      `}</style>
    </div>
  );
}
