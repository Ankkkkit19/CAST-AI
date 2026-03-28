import { useState } from 'react'
import { useA11y } from '../context/AccessibilityContext'
import './CoursePage.css'

const ACTIVITIES = [
  { id: 1, title: 'Finger Painting', desc: 'Use your fingers to create shapes and colors. Great for motor skills!', tip: 'Try primary colors: red, blue, yellow' },
  { id: 2, title: 'Sound Art', desc: 'Create art that makes sounds. Tap, scratch, or blow to hear different tones.', tip: 'Explore rhythm and melody through touch' },
  { id: 3, title: 'Texture Collage', desc: 'Collect rough, smooth, soft, and bumpy materials. Glue them into a collage.', tip: 'Describe each texture as you work' },
  { id: 4, title: 'Movement Drawing', desc: 'Move your whole body to draw. Big arm circles, small finger dots!', tip: 'Use large paper on the wall or floor' },
  { id: 5, title: 'Color Stories', desc: 'Pick a color. Draw or describe everything that color makes you feel.', tip: 'Red = energy, Blue = calm, Green = nature' },
]

export default function CreativeArts({ setPage }) {
  const { settings } = useA11y()
  const [selected, setSelected] = useState(0)
  const activity = ACTIVITIES[selected]

  const speak = (text) => {
    if (!settings.tts || !window.speechSynthesis) return
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(text)
    utt.rate = 0.9
    window.speechSynthesis.speak(utt)
  }

  return (
    <main id="main-content" className="course-page">
      <div className="container">
        <div className="course-page__header fade-in-up">
          <div>
            <h1 className="course-page__title">🎨 Creative Arts & Expression</h1>
            <p className="course-page__subtitle">Fully accessible art activities for all learners</p>
          </div>
          <button className="btn btn-ghost" onClick={() => setPage('dashboard')}>← Back to Courses</button>
        </div>

        <div className="course-page__main course-page__main--single">
          <div className="course-page__content glass-card fade-in-up delay-1" style={{ maxWidth: '600px', margin: '0 auto' }}>
            <h2 className="course-page__exercise-title">{activity.title}</h2>
            <p className="course-page__exercise-desc">{activity.desc}</p>
            <p className="course-page__tip">💡 {activity.tip}</p>
            <button className="btn btn-primary" onClick={() => speak(activity.desc + ' ' + activity.tip)}>
              🔊 Hear Activity
            </button>
          </div>
        </div>

        <div className="course-page__pills fade-in-up delay-2">
          {ACTIVITIES.map((a, i) => (
            <button
              key={a.id}
              className={`btn btn-sm ${selected === i ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setSelected(i)}
            >
              {a.title}
            </button>
          ))}
        </div>
      </div>
    </main>
  )
}
