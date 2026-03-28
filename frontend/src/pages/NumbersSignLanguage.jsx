import { useState, useEffect } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import { useAuth } from '../context/AuthContext'
import { saveProgress } from '../services/progressService'
import './CoursePage.css'

const NUMBERS_1_20 = [
  { num: 1, sign: 'Index finger up', tip: 'Like pointing to the sky!' },
  { num: 2, sign: 'Index + middle finger up (peace sign)', tip: 'Spread them apart for 2!' },
  { num: 3, sign: 'Index, middle, ring fingers up', tip: 'Three fingers pointing up' },
  { num: 4, sign: 'Thumb tucked, 4 fingers up', tip: 'Like the letter B but fingers spread' },
  { num: 5, sign: 'All 5 fingers spread', tip: 'High five! Open hand' },
  { num: 6, sign: 'Thumb touches pinky, other 4 up', tip: 'Thumb and pinky make a hook' },
  { num: 7, sign: 'Thumb touches ring finger, others up', tip: 'Move thumb down one finger' },
  { num: 8, sign: 'Thumb touches middle finger', tip: 'Thumb connects to middle' },
  { num: 9, sign: 'Thumb touches index finger', tip: 'Almost like the letter F' },
  { num: 10, sign: 'Thumbs up or shake closed fist', tip: 'Celebrate ten!' },
  { num: 11, sign: 'Index finger flicks up twice', tip: 'Quick flick for eleven' },
  { num: 12, sign: '1 and 2 combined', tip: 'Sign 1 then 2 in sequence' },
  { num: 13, sign: '1 and 3 combined', tip: 'Sign 1 then 3' },
  { num: 14, sign: '1 and 4 combined', tip: 'Sign 1 then 4' },
  { num: 15, sign: 'Closed fist, thumb out', tip: 'Fist with thumb extended' },
  { num: 16, sign: 'A handshape, twist at wrist', tip: 'Letter A with a twist' },
  { num: 17, sign: 'Q handshape, twist', tip: 'Q shape with movement' },
  { num: 18, sign: 'R handshape, twist', tip: 'R shape with twist' },
  { num: 19, sign: 'S handshape, twist', tip: 'S shape with movement' },
  { num: 20, sign: 'Grasp and release twice', tip: 'Pinch and release for twenty' },
]

export default function NumbersSignLanguage({ setPage }) {
  const { settings } = useA11y()
  const { currentUser } = useAuth()
  const [stepIndex, setStepIndex] = useState(0)
  const [seen, setSeen] = useState(new Set([0]))
  const selectedNum = NUMBERS_1_20[stepIndex]?.num ?? 1

  // Save progress whenever a new number is visited
  useEffect(() => {
    if (currentUser) {
      const newSeen = new Set(seen);
      newSeen.add(stepIndex);
      setSeen(newSeen);
      const pct = Math.round((newSeen.size / 20) * 100);
      saveProgress(currentUser.uid, 'numbers', {
        completedCount: newSeen.size,
        currentNumber: NUMBERS_1_20[stepIndex]?.num,
        currentStep: stepIndex + 1,
        total: 20,
        percent: pct,
        email: currentUser.email,
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepIndex]);

  const speak = (text) => {
    if (!settings.tts || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    
    setTimeout(() => {
      const utt = new SpeechSynthesisUtterance(String(text));
      utt.rate = 0.9;
      
      window._verbalUtterances = window._verbalUtterances || [];
      window._verbalUtterances.push(utt);
      
      utt.onend = () => {
         window._verbalUtterances = window._verbalUtterances.filter(u => u !== utt);
      };
      
      window.speechSynthesis.speak(utt);
    }, 50);
  }

  const goToStep = (idx) => {
    const next = Math.max(0, Math.min(NUMBERS_1_20.length - 1, idx))
    setStepIndex(next)
    speak(NUMBERS_1_20[next].num)
  }

  const item = NUMBERS_1_20.find(n => n.num === selectedNum)

  return (
    <main id="main-content" className="course-page">
      <div className="container">
        <div className="course-page__header fade-in-up">
          <div>
            <h1 className="course-page__title">🔢 Numbers in Sign Language</h1>
            <p className="course-page__subtitle">Step-by-step 1–20. Use Prev/Next or tap any number.</p>
          </div>
          <button className="btn btn-ghost" onClick={() => setPage('dashboard')}>← Back to Courses</button>
        </div>

        {/* Step navigation */}
        <div className="course-page__step-nav glass-card fade-in-up">
          <span className="course-page__step-label">Step {stepIndex + 1} of 20</span>
          <div className="course-page__step-buttons">
            <button className="btn btn-ghost btn-sm" onClick={() => goToStep(stepIndex - 1)} disabled={stepIndex === 0}>← Previous</button>
            <button className="btn btn-primary btn-sm" onClick={() => goToStep(stepIndex + 1)} disabled={stepIndex === 19}>Next →</button>
          </div>
        </div>

        <div className="course-page__main">
          <div className="course-page__display glass-card fade-in-up delay-1">
            <div className="course-page__big-value">{selectedNum}</div>
            <p className="course-page__hear" onClick={() => speak(selectedNum)} role="button" tabIndex="0">
              🔊 Click to hear "{selectedNum}"
            </p>
          </div>
          <div className="course-page__content glass-card fade-in-up delay-2">
            <h3>How to sign {selectedNum}</h3>
            <p className="course-page__sign-desc">{item?.sign}</p>
            <p className="course-page__tip">💡 {item?.tip}</p>
            <button className="btn btn-primary" onClick={() => setPage('detector')}>
              🤝 Practice in Sign Detector
            </button>
          </div>
        </div>

        <div className="course-page__grid-section fade-in-up delay-3">
          <p className="course-page__grid-label">Step {stepIndex + 1} of 20 — or tap any number:</p>
          <div className="course-page__grid">
            {NUMBERS_1_20.map((n, i) => (
              <button
                key={n.num}
                className={`course-page__cell ${selectedNum === n.num ? 'active' : ''} ${i < stepIndex ? 'done' : ''}`}
                onClick={() => goToStep(i)}
                aria-label={`Number ${n.num}`}
              >
                {n.num}
              </button>
            ))}
          </div>
        </div>
      </div>
    </main>
  )
}
