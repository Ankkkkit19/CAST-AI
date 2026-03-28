import { useState } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import './CoursePage.css'

const EXERCISES = [
  { id: 1, title: 'Vowel Warm-Up', desc: 'Practice A, E, I, O, U clearly. Hold each sound for 2 seconds.', icon: '🗣️' },
  { id: 2, title: 'Tongue Twisters', desc: '"She sells seashells" — Say it 3 times slowly, then faster!', icon: '🌀' },
  { id: 3, title: 'Breath Support', desc: 'Take a deep breath. Exhale slowly while counting to 10.', icon: '💨' },
  { id: 4, title: 'Consonant Clarity', desc: 'Focus on P, T, K, B, D, G. Repeat each 5 times clearly.', icon: '🔤' },
  { id: 5, title: 'Sentence Building', desc: 'Say: "I am learning to speak clearly." Repeat 3 times.', icon: '📝' },
]

export default function SpeechTherapy({ setPage }) {
  const { settings } = useA11y()
  const [current, setCurrent] = useState(0)
  const exercise = EXERCISES[current]

  const speak = (text) => {
    if (!settings.tts || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(text)
    utt.rate = 0.85
    window.speechSynthesis.speak(utt)
  }

  return (
    <main id="main-content" className="course-page">
      <div className="container">
        <div className="course-page__header fade-in-up">
          <div>
            <h1 className="course-page__title">🗣️ Speech Therapy Exercises</h1>
            <p className="course-page__subtitle">AI-guided exercises to improve pronunciation and confidence</p>
          </div>
          <button className="btn btn-ghost" onClick={() => setPage('dashboard')}>← Back to Courses</button>
        </div>

        <div className="course-page__main course-page__main--single">
          <div className="course-page__content glass-card fade-in-up delay-1" style={{ maxWidth: '600px', margin: '0 auto' }}>
            <div className="course-page__exercise-icon">{exercise.icon}</div>
            <h2 className="course-page__exercise-title">{exercise.title}</h2>
            <p className="course-page__exercise-desc">{exercise.desc}</p>
            <div className="course-page__exercise-actions">
              <button className="btn btn-primary" onClick={() => speak(exercise.desc)}>
                🔊 Hear Example
              </button>
              <div className="course-page__nav-btns">
                <button
                  className="btn btn-ghost"
                  disabled={current === 0}
                  onClick={() => setCurrent(c => Math.max(0, c - 1))}
                >
                  ← Previous
                </button>
                <button
                  className="btn btn-primary"
                  disabled={current === EXERCISES.length - 1}
                  onClick={() => setCurrent(c => Math.min(EXERCISES.length - 1, c + 1))}
                >
                  Next →
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="course-page__pills fade-in-up delay-2">
          {EXERCISES.map((ex, i) => (
            <button
              key={ex.id}
              className={`btn btn-sm ${current === i ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setCurrent(i)}
            >
              {ex.icon} {ex.title}
            </button>
          ))}
        </div>
      </div>
    </main>
  )
}
